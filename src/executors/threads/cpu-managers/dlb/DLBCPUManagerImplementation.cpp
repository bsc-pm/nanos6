/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <ctime>
#include <dlb.h>

#include "DLBCPUActivation.hpp"
#include "DLBCPUManagerImplementation.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/HardwareInfo.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


void DLBCPUManagerImplementation::preinitialize(__attribute__((unused)) bool dlbEnabled)
{
	_finishedCPUInitialization = false;

	// Set the mask of the process
	int rc = sched_getaffinity(0, sizeof(cpu_set_t), &_cpuMask);
	FatalErrorHandler::handle(rc, " when retrieving the affinity of the process");

	// Get the number of NUMA nodes
	const size_t numNUMANodes = HardwareInfo::getMemoryPlaceCount(nanos6_device_t::nanos6_host_device);
	_NUMANodeMask.resize(numNUMANodes);

	// Get information about the available CPUs in the system
	HostInfo *hostInfo = (HostInfo *) HardwareInfo::getDeviceInfo(nanos6_device_t::nanos6_host_device);
	std::vector<ComputePlace *> const &systemCPUs = hostInfo->getComputePlaces();
	size_t numCPUs = systemCPUs.size();


	//    TASKFOR GROUPS    //

	// Whether the taskfor group envvar already has a value
	bool taskforGroupsSetByUser = _taskforGroups.isPresent();

	// FIXME-TODO: Find in the future an appropriate mechanism to control
	// taskfor groups for external CPUs
	size_t defaultTaskforGroups = _taskforGroups;
	_taskforGroups.setValue(1);
	if (taskforGroupsSetByUser && defaultTaskforGroups != 1) {
		FatalErrorHandler::warnIf(
			true,
			"DLB enabled, using 1 group of ", numCPUs,
			" CPUs instead of the set value"
		);
	}


	//    CPU MANAGER STRUCTURES    //

	// Initialize the vectors of CPUs and mappings
	_cpus.resize(numCPUs);
	for (size_t i = 0; i < numNUMANodes; ++i) {
		_NUMANodeMask[i].resize(numCPUs);
	}

	// Set attributes for all CPUs
	for (size_t i = 0; i < numCPUs; ++i) {
		// Place the CPU in the vectors
		CPU *cpu = (CPU *) systemCPUs[i];
		assert(cpu != nullptr);

		_cpus[i] = cpu;
		_NUMANodeMask[cpu->getNumaNodeId()][i] = true;

		// Set the virtual id (identical to system id)
		cpu->setIndex(i);

		// FIXME-TODO: Since we cannot control when external CPUs are returned,
		// we set all CPUs to the same group so regardless of the group, there
		// will be available CPUs to execute any taskfor
		cpu->setGroupId(0);

		// If the CPU is not owned by this process, mark it as such
		if (!CPU_ISSET(i, &_cpuMask)) {
			cpu->setOwned(false);
		}
	}

	// Set all CPUs as not idle. This is used in the shutdown process
	_idleCPUs.resize(numCPUs);
	_idleCPUs.reset();

	CPUManagerInterface::reportInformation(numCPUs, numNUMANodes);


	//    DLB RELATED    //

	// NOTE: We use the sync (or polling) version of the library. This means
	// that when a call to DLB returns, all the required actions have been
	// taken (i.e. all the callbacks have been triggered before returning)
	assert(_cpus.size() > 0);
	int ret = DLB_Init(_cpus.size(), &_cpuMask, _dlbOptions);
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

void DLBCPUManagerImplementation::initialize()
{
	for (size_t id = 0; id < _cpus.size(); ++id) {
		CPU *cpu = _cpus[id];
		assert(cpu != nullptr);

		// If this CPU is owned by this process, initialize it if needed
		if (cpu->isOwned()) {
			// This should always work
			bool worked = cpu->initializeIfNeeded();
			assert(worked);

			if (worked) {
				WorkerThread *initialThread = ThreadManager::createWorkerThread(cpu);
				assert(initialThread != nullptr);

				initialThread->resume(cpu, true);
			}
		}
	}

	_finishedCPUInitialization = true;
}

void DLBCPUManagerImplementation::shutdownPhase1()
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

void DLBCPUManagerImplementation::shutdownPhase2()
{
	// Shutdown DLB
	// ret != DLB_SUCCESS means it was not initialized (should never occur)
	__attribute__((unused)) int ret = DLB_Finalize();
	assert(ret == DLB_SUCCESS);
}

void DLBCPUManagerImplementation::executeCPUManagerPolicy(ComputePlace *cpu, CPUManagerPolicyHint hint, size_t numTasks)
{
	// NOTE This policy works as follows:
	// - First, if the CPU is an acquired one, check if we must return it
	// - Then if the hint is IDLE_CANDIDATE we try to lend the CPU if
	//   possible
	// - If the hint is ADDED_TASKS, we try to reclaim as many lent CPUs
	//   as tasks were added or acquire new ones
	// - If the hint is HANDLE_TASKFOR, we try to reclaim all CPUs that can
	//   collaborate in the taskfor
	CPU *currentCPU = (CPU *) cpu;
	if (hint == IDLE_CANDIDATE) {
		assert(currentCPU != nullptr);

		// If we own the CPU lend it; otherwise, check if we must return it
		if (currentCPU->isOwned()) {
			DLBCPUActivation::lendCPU(currentCPU);
		} else {
			DLBCPUActivation::checkIfMustReturnCPU(currentCPU);
		}
	} else if (hint == ADDED_TASKS) {
		assert(numTasks > 0);

		// Try to obtain as many CPUs as tasks were added
		size_t numToObtain = std::min(_cpus.size(), numTasks);
		DLBCPUActivation::acquireCPUs(numToObtain);
	} else { // hint = HANDLE_TASKFOR
		assert(currentCPU != nullptr);

		// Try to reclaim any lent collaborator of the taskfor
		cpu_set_t collaboratorMask = getCollaboratorMask(currentCPU);
		if (CPU_COUNT(&collaboratorMask) > 0) {
			DLBCPUActivation::acquireCPUs(collaboratorMask);
		}
	}
}


/*    CPUACTIVATION BRIDGE    */

CPU::activation_status_t DLBCPUManagerImplementation::checkCPUStatusTransitions(WorkerThread *thread)
{
	return DLBCPUActivation::checkCPUStatusTransitions(thread);
}

bool DLBCPUManagerImplementation::acceptsWork(CPU *cpu)
{
	return DLBCPUActivation::acceptsWork(cpu);
}

bool DLBCPUManagerImplementation::enable(size_t systemCPUId)
{
	return DLBCPUActivation::enable(systemCPUId);
}

bool DLBCPUManagerImplementation::disable(size_t systemCPUId)
{
	return DLBCPUActivation::disable(systemCPUId);
}


/*    IDLE CPUS    */

// NOTE: The following functions should only be used when the runtime is
// shutting down, to allow all threads to shutdown

bool DLBCPUManagerImplementation::cpuBecomesIdle(CPU *cpu, bool inShutdown)
{
	if (inShutdown) {
		assert(cpu != nullptr);

		const int index = cpu->getIndex();
		_idleCPUsLock.lock();

		// The CPU should not be marked as idle
		assert(!_idleCPUs[index]);

		_idleCPUs[index] = true;
		_idleCPUsLock.unlock();
	}

	return inShutdown;
}

CPU *DLBCPUManagerImplementation::getIdleCPU(bool inShutdown)
{
	if (inShutdown) {
		_idleCPUsLock.lock();
		boost::dynamic_bitset<>::size_type idleCPU = _idleCPUs.find_first();
		if (idleCPU != boost::dynamic_bitset<>::npos) {
			_idleCPUs[idleCPU] = false;
			_idleCPUsLock.unlock();

			return _cpus[idleCPU];
		}
		_idleCPUsLock.unlock();
	}

	return nullptr;
}

