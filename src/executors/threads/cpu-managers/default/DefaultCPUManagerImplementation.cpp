/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "DefaultCPUActivation.hpp"
#include "DefaultCPUManagerImplementation.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/cpu-managers/default/policies/BusyPolicy.hpp"
#include "executors/threads/cpu-managers/default/policies/IdlePolicy.hpp"

#include <InstrumentComputePlaceManagement.hpp>
#include <Monitoring.hpp>


boost::dynamic_bitset<> DefaultCPUManagerImplementation::_idleCPUs;
SpinLock DefaultCPUManagerImplementation::_idleCPUsLock;
size_t DefaultCPUManagerImplementation::_numIdleCPUs;
std::vector<size_t> DefaultCPUManagerImplementation::_systemToVirtualCPUId;


/*    CPUMANAGER    */

void DefaultCPUManagerImplementation::preinitialize()
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

	// Create the chosen policy for this CPUManager
	std::string policyValue = _policyChosen.getValue();
	if (policyValue == "default" || policyValue == "idle") {
		_cpuManagerPolicy = new IdlePolicy(numCPUs);
	} else if (policyValue == "busy") {
		_cpuManagerPolicy = new BusyPolicy();
	} else {
		FatalErrorHandler::failIf(
			true, "Unexistent '", policyValue, "' CPU Manager Policy"
		);
	}
	assert(_cpuManagerPolicy != nullptr);

	// Find the appropriate value for taskfor groups
	refineTaskforGroups(numCPUs, numNUMANodes);

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

	const size_t numCPUsPerTaskforGroup = getNumCPUsPerTaskforGroup();
	assert(numCPUsPerTaskforGroup > 0);

	// Initialize each CPU's fields
	size_t virtualCPUId = 0;
	for (size_t i = 0; i < numCPUs; ++i) {
		CPU *cpu = (CPU *) cpus[i];
		if (CPU_ISSET(cpu->getSystemCPUId(), &_cpuMask)) {
			// To compute the groupId we need the CPU's hwloc logical_index
			// Before setting the virtualID, use the current id for the groupId
			size_t groupId = cpu->getIndex() / numCPUsPerTaskforGroup;
			assert(groupId <= numCPUs);

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

	reportInformation(numSystemCPUs, numNUMANodes);

	// Initialize idle CPU structures
	_idleCPUs.resize(numAvailableCPUs);
	_idleCPUs.reset();
	_numIdleCPUs = 0;
}

void DefaultCPUManagerImplementation::initialize()
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

void DefaultCPUManagerImplementation::shutdownPhase1()
{
	// Notify all CPUs that the runtime is shutting down
	for (size_t id = 0; id < _cpus.size(); ++id) {
		DefaultCPUActivation::shutdownCPU(_cpus[id]);
	}
}


/*    CPUACTIVATION BRIDGE    */

CPU::activation_status_t DefaultCPUManagerImplementation::checkCPUStatusTransitions(WorkerThread *thread)
{
	return DefaultCPUActivation::checkCPUStatusTransitions(thread);
}

bool DefaultCPUManagerImplementation::acceptsWork(CPU *cpu)
{
	return DefaultCPUActivation::acceptsWork(cpu);
}

bool DefaultCPUManagerImplementation::enable(size_t systemCPUId)
{
	return DefaultCPUActivation::enable(systemCPUId);
}

bool DefaultCPUManagerImplementation::disable(size_t systemCPUId)
{
	return DefaultCPUActivation::disable(systemCPUId);
}


/*    IDLE MECHANISM    */

bool DefaultCPUManagerImplementation::cpuBecomesIdle(CPU *cpu)
{
	assert(cpu != nullptr);

	const int index = cpu->getIndex();

	_idleCPUsLock.lock();

	// Before idling the CPU, check if there truly aren't any tasks ready
	// NOTE: This is a workaround to solve the race condition between adding
	// tasks and idling CPUs; i.e. it may happen that before a CPU is idled,
	// tasks are added in the scheduler and that CPU may never have the chance
	// to wake up and execute these newly added tasks
	if (Scheduler::hasAvailableWork(cpu)) {
		// If there are ready tasks, release the lock and do not idle the CPU
		_idleCPUsLock.unlock();

		return false;
	}

	// Mark the CPU as idle
	Monitoring::cpuBecomesIdle(index);
	Instrument::suspendingComputePlace(cpu->getInstrumentationId());
	_idleCPUs[index] = true;
	++_numIdleCPUs;
	assert(_numIdleCPUs <= _cpus.size());

	_idleCPUsLock.unlock();

	return true;
}

CPU *DefaultCPUManagerImplementation::getIdleCPU()
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

size_t DefaultCPUManagerImplementation::getIdleCPUs(
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

void DefaultCPUManagerImplementation::getIdleCollaborators(
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
