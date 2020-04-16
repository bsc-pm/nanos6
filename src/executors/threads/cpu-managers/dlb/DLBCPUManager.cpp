/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <ctime>
#include <dlb.h>

#include "DLBCPUActivation.hpp"
#include "DLBCPUManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "executors/threads/cpu-managers/dlb/policies/GreedyPolicy.hpp"
#include "executors/threads/cpu-managers/dlb/policies/LeWIPolicy.hpp"
#include "hardware/HardwareInfo.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


boost::dynamic_bitset<> DLBCPUManager::_shutdownCPUs;
SpinLock DLBCPUManager::_shutdownCPUsLock;


void DLBCPUManager::preinitialize()
{
	_finishedCPUInitialization = false;

	// Retreive the CPU mask of this process
	int rc = sched_getaffinity(0, sizeof(cpu_set_t), &_cpuMask);
	FatalErrorHandler::handle(
		rc, " when retrieving the affinity of the process"
	);

	// Get the number of NUMA nodes and a list of all available CPUs
	nanos6_device_t hostDevice = nanos6_device_t::nanos6_host_device;
	const size_t numNUMANodes = HardwareInfo::getMemoryPlaceCount(hostDevice);
	HostInfo *hostInfo = ((HostInfo *) HardwareInfo::getDeviceInfo(hostDevice));
	assert(hostInfo != nullptr);

	std::vector<ComputePlace *> const &cpus = hostInfo->getComputePlaces();
	size_t numCPUs = cpus.size();
	assert(numCPUs > 0);

	// Create the chosen policy for this CPUManager
	std::string policyValue = _policyChosen.getValue();
	if (policyValue == "default" || policyValue == "lewi") {
		_cpuManagerPolicy = new LeWIPolicy(numCPUs);
	} else if (policyValue == "greedy") {
		_cpuManagerPolicy = new GreedyPolicy(numCPUs);
	} else {
		FatalErrorHandler::failIf(
			true, "Unexistent '", policyValue, "' CPU Manager Policy"
		);
	}
	assert(_cpuManagerPolicy != nullptr);


	//    TASKFOR GROUPS    //

	// FIXME-TODO: Find an appropriate mechanism to set the env var
	_taskforGroups.setValue(1);


	//    CPU MANAGER STRUCTURES    //

	// Initialize the vector of CPUs
	_cpus.resize(numCPUs);

	// Initialize each CPU's fields
	bool firstCPUFound = false;
	for (size_t i = 0; i < numCPUs; ++i) {
		// Place the CPU in the vectors
		CPU *cpu = (CPU *) cpus[i];
		assert(cpu != nullptr);

		_cpus[i] = cpu;

		// Set the virtual id (identical to system id)
		cpu->setIndex(i);

		// FIXME-TODO: Since we cannot control when external CPUs are returned,
		// we set all CPUs to the same group so regardless of the group, there
		// will be available CPUs to execute any taskfor
		cpu->setGroupId(0);

		// If the CPU is not owned by this process, mark it as such
		if (!CPU_ISSET(i, &_cpuMask)) {
			cpu->setOwned(false);
		} else if (!firstCPUFound) {
			_firstCPUId = i;
			firstCPUFound = true;
		}
	}
	assert(firstCPUFound);

	CPUManagerInterface::reportInformation(numCPUs, numNUMANodes);
	if (_taskforGroupsReportEnabled) {
		CPUManagerInterface::reportTaskforGroupsInfo();
	}

	// All CPUs are unavailable for the shutdown process at the start
	_shutdownCPUs.resize(numCPUs);
	_shutdownCPUs.reset();


	//    DLB RELATED    //

	// NOTE: We use the sync (or polling) version of the library. This means
	// that when a call to DLB returns, all the required actions have been
	// taken (i.e. all the callbacks have been triggered before returning)
	int ret = DLB_Init(numCPUs, &_cpuMask, "--lewi --quiet=yes");
	if (ret == DLB_ERR_PERM) {
		FatalErrorHandler::failIf(
			true,
			"The current CPU mask collides with another process' mask\n",
			"Original error code while initializing DLB: ", ret
		);
	} else if (ret != DLB_SUCCESS) {
		FatalErrorHandler::failIf(
			true,
			"Error code ", ret, " while initializing DLB"
		);
	}

	// Prepare callbacks to enable/disable CPUs from DLB
	ret = DLB_CallbackSet(
		dlb_callback_enable_cpu,
		(dlb_callback_t)DLBCPUActivation::dlbEnableCallback,
		nullptr
	);
	if (ret == DLB_SUCCESS) {
		ret = DLB_CallbackSet(
			dlb_callback_disable_cpu,
			(dlb_callback_t)DLBCPUActivation::dlbDisableCallback,
			nullptr
		);
	}
	FatalErrorHandler::failIf(
		ret != DLB_SUCCESS,
		"Error code ", ret, " while registering DLB callbacks"
	);
}

void DLBCPUManager::initialize()
{
	for (size_t id = 0; id < _cpus.size(); ++id) {
		CPU *cpu = _cpus[id];
		assert(cpu != nullptr);

		// If this CPU is owned by this process, initialize it if needed
		if (cpu->isOwned()) {
			__attribute__((unused)) bool worked = cpu->initializeIfNeeded();
			assert(worked);

			WorkerThread *initialThread = ThreadManager::createWorkerThread(cpu);
			assert(initialThread != nullptr);

			initialThread->resume(cpu, true);
		}
	}

	_finishedCPUInitialization = true;
}

void DLBCPUManager::shutdownPhase1()
{
	CPU *cpu;
	CPU::activation_status_t status;
	const timespec delay = {0, 100000};

	// Phase 1.1 - Signal the shutdown to all CPUs
	for (size_t id = 0; id < _cpus.size(); ++id) {
		cpu = _cpus[id];
		assert(cpu != nullptr);

		status = cpu->getActivationStatus();
		assert(status != CPU::shutdown_status && status != CPU::shutting_down_status);

		DLBCPUActivation::shutdownCPU(cpu);
	}

	// Phase 1.2 - Wait until all CPUs are shutdown
	for (size_t id = 0; id < _cpus.size(); ++id) {
		cpu = _cpus[id];
		status = cpu->getActivationStatus();
		while (status != CPU::shutdown_status && status != CPU::uninitialized_status) {
			// Sleep for a short amount of time
			nanosleep(&delay, nullptr);
			status = cpu->getActivationStatus();
		}
	}

	// NOTE: At this point all CPUs are shutting down, threads should
	// progressively see this and add themselves to the shutdown list
}

void DLBCPUManager::shutdownPhase2()
{
	// Shutdown DLB
	// ret != DLB_SUCCESS means it was not initialized (should never occur)
	__attribute__((unused)) int ret = DLB_Finalize();
	assert(ret == DLB_SUCCESS);

	delete _cpuManagerPolicy;

	_cpuManagerPolicy = nullptr;
}

void DLBCPUManager::forcefullyResumeFirstCPU()
{
	// Try to reclaim the CPU (it only happens if it is lent)
	DLBCPUActivation::reclaimCPU(_firstCPUId);
}


/*    CPUACTIVATION BRIDGE    */

CPU::activation_status_t DLBCPUManager::checkCPUStatusTransitions(WorkerThread *thread)
{
	return DLBCPUActivation::checkCPUStatusTransitions(thread);
}

bool DLBCPUManager::acceptsWork(CPU *cpu)
{
	return DLBCPUActivation::acceptsWork(cpu);
}

bool DLBCPUManager::enable(size_t systemCPUId)
{
	return DLBCPUActivation::enable(systemCPUId);
}

bool DLBCPUManager::disable(size_t systemCPUId)
{
	return DLBCPUActivation::disable(systemCPUId);
}
