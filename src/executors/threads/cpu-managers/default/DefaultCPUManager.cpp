/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include "DefaultCPUActivation.hpp"
#include "DefaultCPUManager.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/cpu-managers/default/policies/BusyPolicy.hpp"
#include "executors/threads/cpu-managers/default/policies/IdlePolicy.hpp"
#include "hardware-counters/HardwareCounters.hpp"
#include "scheduling/Scheduler.hpp"

#include <InstrumentComputePlaceManagement.hpp>
#include <Monitoring.hpp>


boost::dynamic_bitset<> DefaultCPUManager::_idleCPUs;
SpinLock DefaultCPUManager::_idleCPUsLock;
size_t DefaultCPUManager::_numIdleCPUs;


/*    CPUMANAGER    */

void DefaultCPUManager::preinitialize()
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

	const std::vector<ComputePlace *> &cpus = hostInfo->getComputePlaces();
	size_t numCPUs = cpus.size();
	assert(numCPUs > 0);

	// Create the chosen policy for this CPUManager
	std::string policyValue = _policyChosen.getValue();
	if (policyValue == "default" || policyValue == "idle") {
		_cpuManagerPolicy = new IdlePolicy(numCPUs);
	} else if (policyValue == "busy") {
		_cpuManagerPolicy = new BusyPolicy();
	} else {
		FatalErrorHandler::fail("Unexistent '", policyValue, "' CPU Manager Policy");
	}
	assert(_cpuManagerPolicy != nullptr);

	// Find the maximum system CPU id
	size_t maxSystemCPUId = 0;
	for (size_t i = 0; i < numCPUs; ++i) {
		const CPU *cpu = (const CPU *) cpus[i];
		assert(cpu != nullptr);

		if (cpu->getSystemCPUId() > maxSystemCPUId) {
			maxSystemCPUId = cpu->getSystemCPUId();
		}
	}

	// Set appropriate sizes for the vector of CPUs and their id maps
	const size_t numSystemCPUs = maxSystemCPUId + 1;
	const size_t numAvailableCPUs = CPU_COUNT(&_cpuMask);
	_cpus.resize(numAvailableCPUs);
	_systemToVirtualCPUId.resize(numSystemCPUs);

	// Find the appropriate value for taskfor groups
	std::vector<int> availableNUMANodes(numNUMANodes, 0);
	for (size_t i = 0; i < numCPUs; i++) {
		CPU *cpu = (CPU *) cpus[i];
		assert(cpu != nullptr);

		if (CPU_ISSET(cpu->getSystemCPUId(), &_cpuMask)) {
			size_t NUMANodeId = cpu->getNumaNodeId();
			availableNUMANodes[NUMANodeId]++;
		}
	}

	size_t numValidNUMANodes = 0;
	for (size_t i = 0; i < numNUMANodes; i++) {
		if (availableNUMANodes[i] > 0) {
			numValidNUMANodes++;
		}
	}
	refineTaskforGroups(numAvailableCPUs, numValidNUMANodes);

	// Initialize each CPU's fields
	size_t groupId = 0;
	size_t virtualCPUId = 0;
	size_t numCPUsPerTaskforGroup = numAvailableCPUs / getNumTaskforGroups();
	assert(numCPUsPerTaskforGroup > 0);

	for (size_t i = 0; i < numCPUs; ++i) {
		CPU *cpu = (CPU *) cpus[i];
		if (CPU_ISSET(cpu->getSystemCPUId(), &_cpuMask)) {
			// Check if this CPU goes into another group
			if (numCPUsPerTaskforGroup == 0) {
				numCPUsPerTaskforGroup = (numAvailableCPUs / getNumTaskforGroups()) - 1;
				++groupId;
			} else {
				--numCPUsPerTaskforGroup;
			}

			cpu->setIndex(virtualCPUId);
			cpu->setGroupId(groupId);
			_cpus[virtualCPUId] = cpu;
			++virtualCPUId;
		} else {
			cpu->setIndex((unsigned int) ~0UL);
		}
		_systemToVirtualCPUId[cpu->getSystemCPUId()] = cpu->getIndex();
	}
	assert(virtualCPUId == numAvailableCPUs);

	// The first CPU is always 0 (virtual identifier)
	_firstCPUId = 0;

	CPUManagerInterface::reportInformation(numSystemCPUs, numNUMANodes);
	if (_taskforGroupsReportEnabled) {
		CPUManagerInterface::reportTaskforGroupsInfo();
	}

	// Initialize idle CPU structures
	_idleCPUs.resize(numAvailableCPUs);
	_idleCPUs.reset();
	_numIdleCPUs = 0;

	// Initialize the virtual CPU for the leader thread
	_leaderThreadCPU = new CPU(numCPUs);
	assert(_leaderThreadCPU != nullptr);
}

void DefaultCPUManager::initialize()
{
	for (size_t id = 0; id < _cpus.size(); ++id) {
		CPU *cpu = _cpus[id];
		assert(cpu != nullptr);

		__attribute__((unused)) bool worked = cpu->initializeIfNeeded();
		assert(worked);

		WorkerThread *initialThread = ThreadManager::createWorkerThread(cpu);
		assert(initialThread != nullptr);

		initialThread->resume(cpu, true);
	}

	_finishedCPUInitialization = true;
}

void DefaultCPUManager::shutdownPhase1()
{
	// Notify all CPUs that the runtime is shutting down
	for (size_t id = 0; id < _cpus.size(); ++id) {
		DefaultCPUActivation::shutdownCPU(_cpus[id]);
	}
}

void DefaultCPUManager::forcefullyResumeFirstCPU()
{
	bool resumed = false;

	_idleCPUsLock.lock();

	if (_idleCPUs[_firstCPUId]) {
		_idleCPUs[_firstCPUId] = false;
		assert(_numIdleCPUs > 0);

		--_numIdleCPUs;

		// Since monitoring works with system ids, translate the ID
		Monitoring::cpuBecomesActive(_cpus[_firstCPUId]->getSystemCPUId());
		resumed = true;
	}

	_idleCPUsLock.unlock();

	if (resumed) {
		assert(_cpus[_firstCPUId] != nullptr);
		ThreadManager::resumeIdle(_cpus[_firstCPUId]);
	}
}


/*    CPUACTIVATION BRIDGE    */

CPU::activation_status_t DefaultCPUManager::checkCPUStatusTransitions(WorkerThread *thread)
{
	return DefaultCPUActivation::checkCPUStatusTransitions(thread);
}

bool DefaultCPUManager::acceptsWork(CPU *cpu)
{
	return DefaultCPUActivation::acceptsWork(cpu);
}

bool DefaultCPUManager::enable(size_t systemCPUId)
{
	return DefaultCPUActivation::enable(systemCPUId);
}

bool DefaultCPUManager::disable(size_t systemCPUId)
{
	return DefaultCPUActivation::disable(systemCPUId);
}


/*    IDLE MECHANISM    */

bool DefaultCPUManager::cpuBecomesIdle(CPU *cpu)
{
	assert(cpu != nullptr);

	const int index = cpu->getIndex();

	std::lock_guard<SpinLock> guard(_idleCPUsLock);

	// If there is no CPU serving tasks in the scheduler,
	// abort the idle process of this CPU and go back
	// to the scheduling part. Notice that this check
	// must be inside the exclusive section covered
	// by the idle mutex
	if (!Scheduler::isServingTasks()) {
		return false;
	}

	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);

	HardwareCounters::updateRuntimeCounters();
	Monitoring::cpuBecomesIdle(index);
	Instrument::threadWillSuspend(currentThread->getInstrumentationId(), cpu->getInstrumentationId());
	Instrument::suspendingComputePlace(cpu->getInstrumentationId());

	// Mark the CPU as idle
	_idleCPUs[index] = true;
	++_numIdleCPUs;
	assert(_numIdleCPUs <= _cpus.size());

	return true;
}

CPU *DefaultCPUManager::getIdleCPU()
{
	std::lock_guard<SpinLock> guard(_idleCPUsLock);

	boost::dynamic_bitset<>::size_type id = _idleCPUs.find_first();
	if (id != boost::dynamic_bitset<>::npos) {
		CPU *cpu = _cpus[id];
		assert(cpu != nullptr);

		Instrument::resumedComputePlace(cpu->getInstrumentationId());
		Monitoring::cpuBecomesActive(id);
		_idleCPUs[id] = false;
		assert(_numIdleCPUs > 0);

		--_numIdleCPUs;

		return cpu;
	} else {
		return nullptr;
	}
}

size_t DefaultCPUManager::getIdleCPUs(
	size_t numCPUs,
	CPU *idleCPUs[]
) {
	size_t numObtainedCPUs = 0;

	std::lock_guard<SpinLock> guard(_idleCPUsLock);

	boost::dynamic_bitset<>::size_type id = _idleCPUs.find_first();
	while (numObtainedCPUs < numCPUs && id != boost::dynamic_bitset<>::npos) {
		CPU *cpu = _cpus[id];
		assert(cpu != nullptr);

		// Signal that the CPU becomes active
		Instrument::resumedComputePlace(cpu->getInstrumentationId());
		Monitoring::cpuBecomesActive(id);
		_idleCPUs[id] = false;

		// Place the CPU in the vector
		idleCPUs[numObtainedCPUs] = cpu;
		++numObtainedCPUs;

		// Iterate to the next idle CPU
		id = _idleCPUs.find_next(id);
	}

	// Decrease the counter of idle CPUs by the obtained amount
	assert(_numIdleCPUs >= numObtainedCPUs);
	_numIdleCPUs -= numObtainedCPUs;

	return numObtainedCPUs;
}

void DefaultCPUManager::getIdleCollaborators(
	std::vector<CPU *> &idleCPUs,
	ComputePlace *cpu
) {
	assert(cpu != nullptr);

	size_t numObtainedCollaborators = 0;
	size_t groupId = ((CPU *) cpu)->getGroupId();

	std::lock_guard<SpinLock> guard(_idleCPUsLock);
	boost::dynamic_bitset<>::size_type id = _idleCPUs.find_first();
	while (id != boost::dynamic_bitset<>::npos) {
		CPU *collaborator = _cpus[id];
		assert(collaborator != nullptr);

		if (groupId == collaborator->getGroupId()) {
			// Signal that the CPU becomes active
			Instrument::resumedComputePlace(collaborator->getInstrumentationId());
			Monitoring::cpuBecomesActive(id);
			_idleCPUs[id] = false;

			// Place the CPU in the vector
			idleCPUs.push_back(collaborator);
			++numObtainedCollaborators;
		}

		// Iterate to the next idle CPU
		id = _idleCPUs.find_next(id);
	}

	// Decrease the counter of idle CPUs by the obtained amount
	assert(_numIdleCPUs >= numObtainedCollaborators);
	_numIdleCPUs -= numObtainedCollaborators;
}


