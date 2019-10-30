/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include <config.h>

#include "CPUManagerInterface.hpp"
#include "ThreadManager.hpp"
#include "WorkerThread.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "scheduling/Scheduler.hpp"
#include "system/RuntimeInfo.hpp"

#include <InstrumentComputePlaceManagement.hpp>
#include <Monitoring.hpp>


std::atomic<bool> CPUManagerInterface::_finishedCPUInitialization;
std::vector<boost::dynamic_bitset<>> CPUManagerInterface::_NUMANodeMask;
std::vector<size_t> CPUManagerInterface::_systemToVirtualCPUId;
std::vector<CPU *> CPUManagerInterface::_cpus;
cpu_set_t CPUManagerInterface::_cpuMask;
SpinLock CPUManagerInterface::_idleCPUsLock;
size_t CPUManagerInterface::_numIdleCPUs;
boost::dynamic_bitset<> CPUManagerInterface::_idleCPUs;
EnvironmentVariable<size_t> CPUManagerInterface::_taskforGroups("NANOS6_TASKFOR_GROUPS", 1);


namespace cpumanager_internals {
	static inline std::string maskToRegionList(boost::dynamic_bitset<> const &mask, size_t size)
	{
		std::ostringstream oss;

		int start = -1;
		int end = -1;
		bool first = true;
		for (size_t i = 0; i < size + 1; i++) {
			if ((i < size) && mask[i]) {
				if (start == -1) {
					start = i;
				}
				end = i;
			} else if (end >= 0) {
				if (first) {
					first = false;
				} else {
					oss << ",";
				}
				if (end == start) {
					oss << start;
				} else {
					oss << start << "-" << end;
				}
				start = -1;
				end = -1;
			}
		}

		return oss.str();
	}
}


/*    CPU MANAGER    */

void CPUManagerInterface::reportInformation(size_t numSystemCPUs, size_t numNUMANodes)
{
	boost::dynamic_bitset<> processCPUMask(numSystemCPUs);

	std::vector<boost::dynamic_bitset<>> NUMANodeSystemMask(numNUMANodes);
	for (size_t i = 0; i < numNUMANodes; ++i) {
		NUMANodeSystemMask[i].resize(numSystemCPUs);
	}

	for (CPU *cpu : _cpus) {
		assert(cpu != nullptr);

		if (cpu->isOwned()) {
			size_t systemCPUId = cpu->getSystemCPUId();
			processCPUMask[systemCPUId] = true;
			NUMANodeSystemMask[cpu->getNumaNodeId()][systemCPUId] = true;
		}
	}

	RuntimeInfo::addEntry(
		"initial_cpu_list",
		"Initial CPU List",
		cpumanager_internals::maskToRegionList(processCPUMask, numSystemCPUs)
	);
	for (size_t i = 0; i < numNUMANodes; ++i) {
		std::ostringstream oss, oss2;

		oss << "numa_node_" << i << "_cpu_list";
		oss2 << "NUMA Node " << i << " CPU List";
		std::string cpuRegionList = cpumanager_internals::maskToRegionList(NUMANodeSystemMask[i], numSystemCPUs);

		RuntimeInfo::addEntry(oss.str(), oss2.str(), cpuRegionList);
	}
}

void CPUManagerInterface::preinitialize()
{
	_finishedCPUInitialization = false;

	int rc = sched_getaffinity(0, sizeof(cpu_set_t), &_cpuMask);
	FatalErrorHandler::handle(rc, " when retrieving the affinity of the process");

	// Get NUMA nodes
	const size_t numNUMANodes = HardwareInfo::getMemoryPlaceCount(nanos6_device_t::nanos6_host_device);
	_NUMANodeMask.resize(numNUMANodes);

	// Default value for _taskforGroups is one per NUMA node
	bool taskforGroupsSetByUser = _taskforGroups.isPresent();
	if (!taskforGroupsSetByUser) {
		_taskforGroups.setValue(numNUMANodes);
	}

	// Get CPU objects that can run a thread
	std::vector<ComputePlace *> const &cpus = ((HostInfo *) HardwareInfo::getDeviceInfo(nanos6_device_t::nanos6_host_device))->getComputePlaces();
	size_t numCPUs = cpus.size();
	if (numCPUs < _taskforGroups && taskforGroupsSetByUser) {
		FatalErrorHandler::warnIf(
			true,
			"More groups requested than available CPUs. ",
			"Using ", numCPUs, " groups of 1 CPU each instead"
		);
		_taskforGroups.setValue(numCPUs);
	}

	// Check if the number of taskfor groups is appropriate
	if (_taskforGroups == 0 || numCPUs % _taskforGroups != 0) {
		size_t closestGroups = getClosestGroupNumber(numCPUs, _taskforGroups);
		size_t cpusPerGroup  = numCPUs / closestGroups;
		_taskforGroups.setValue(closestGroups);

		FatalErrorHandler::warnIf(
			_taskforGroups == 0,
			"0 groups requested, invalid number. ",
			"Using ", closestGroups, " of ", cpusPerGroup, " CPUs each instead"
		);
		FatalErrorHandler::warnIf(
			_taskforGroups != 0 && numCPUs % _taskforGroups != 0,
			_taskforGroups, " groups requested. ",
			"The number of CPUs is not divisiable by the number of groups. ",
			"Using ", closestGroups, " of ", cpusPerGroup, " CPUs each instead"
		);
	}
	assert(_taskforGroups <= numCPUs && numCPUs % _taskforGroups == 0);

	size_t maxSystemCPUId = 0;
	for (size_t i = 0; i < numCPUs; ++i) {
		const CPU *cpu = (const CPU *) cpus[i];
		assert(cpu != nullptr);

		if (cpu->getSystemCPUId() > maxSystemCPUId) {
			maxSystemCPUId = cpu->getSystemCPUId();
		}
	}

	const size_t numSystemCPUs = maxSystemCPUId + 1;
	const size_t numAvailableCPUs = CPU_COUNT(&_cpuMask);
	_cpus.resize(numAvailableCPUs);
	_systemToVirtualCPUId.resize(numSystemCPUs);

	for (size_t i = 0; i < numNUMANodes; ++i) {
		_NUMANodeMask[i].resize(numAvailableCPUs);
	}

	size_t virtualCPUId = 0;
	for (size_t i = 0; i < numCPUs; ++i) {
		CPU *cpu = (CPU *) cpus[i];
		assert(cpu != nullptr);

		if (CPU_ISSET(cpu->getSystemCPUId(), &_cpuMask)) {
			// We need the hwloc logical_index to compute the groupId. However,
			// that index is overwritten, so this is the last place where we
			// still have the hwloc logical_index, so we compute the groupId
			// here and set it as member of CPU
			size_t groupId = cpu->getIndex() / getNumCPUsPerTaskforGroup();
			assert(groupId <= numCPUs);

			cpu->setGroupId(groupId);
			cpu->setIndex(virtualCPUId);
			_cpus[virtualCPUId] = cpu;
			_NUMANodeMask[cpu->getNumaNodeId()][virtualCPUId] = true;
			++virtualCPUId;
		} else {
			cpu->setIndex((unsigned int) ~0UL);
		}
		_systemToVirtualCPUId[cpu->getSystemCPUId()] = cpu->getIndex();
	}
	assert(virtualCPUId == numAvailableCPUs);

	reportInformation(numSystemCPUs, numNUMANodes);

	// Set all CPUs as not idle
	_idleCPUs.resize(numAvailableCPUs);
	_idleCPUs.reset();
	_numIdleCPUs = 0;
}

void CPUManagerInterface::initialize()
{
	for (size_t virtualCPUId = 0; virtualCPUId < _cpus.size(); ++virtualCPUId) {
		if (_cpus[virtualCPUId] != nullptr) {
			CPU *cpu = _cpus[virtualCPUId];
			assert(cpu != nullptr);

			bool worked = cpu->initializeIfNeeded();
			if (worked) {
				WorkerThread *initialThread = ThreadManager::createWorkerThread(cpu);
				initialThread->resume(cpu, true);
			} else {
				// Already initialized?
			}
		}
	}

	_finishedCPUInitialization = true;
}

void CPUManagerInterface::executeCPUManagerPolicy(ComputePlace *cpu, CPUManagerPolicyHint hint, size_t numTasks)
{
	//! NOTE: This policy works as follows:
	//! - CPUs are idled if the hint is IDLE_CANDIDATE and the runtime is not
	//!   shutting down
	//! - Idle CPUs are woken up if the hint is ADDED_TASKS
	//!   - Furthermore, as many CPUs are awaken as tasks are added (at most
	//!     the amount of idle CPUs)
	//! - If the hint is ADDED_TASKFOR, we wake up all the idle collaborators
	if (hint == IDLE_CANDIDATE) {
		assert(cpu != nullptr);

		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		assert(currentThread != nullptr);

		bool cpuIsIdle = cpuBecomesIdle((CPU *) cpu);
		if (cpuIsIdle) {
			// Account this CPU as idle and mark the thread as idle
			Instrument::suspendingComputePlace(cpu->getInstrumentationId());
			Monitoring::cpuBecomesIdle(cpu->getIndex());

			ThreadManager::addIdler(currentThread);
			currentThread->switchTo(nullptr);

			// The thread may have migrated, update the compute place
			cpu = currentThread->getComputePlace();
			assert(cpu != nullptr);

			Instrument::resumedComputePlace(cpu->getInstrumentationId());
			Monitoring::cpuBecomesActive(cpu->getIndex());
		}
	} else if (hint == ADDED_TASKS) {
		assert(numTasks > 0);

		// At most we will obtain as many idle CPUs as the maximum amount
		size_t numCPUsToObtain = std::min(_cpus.size(), numTasks);
		std::vector<CPU *> idleCPUs(numCPUsToObtain, nullptr);

		// Try to get as many idle CPUs as we need
		size_t numCPUsObtained = getIdleCPUs(idleCPUs, numCPUsToObtain);

		// Resume an idle thread for every idle CPU that has awakened
		for (size_t i = 0; i < numCPUsObtained; ++i) {
			assert(idleCPUs[i] != nullptr);
			ThreadManager::resumeIdle(idleCPUs[i]);
		}
	} else { // hint = HANDLE_TASKFOR
		assert(cpu != nullptr);

		std::vector<CPU *> idleCPUs;
		CPUManager::getIdleCollaborators(idleCPUs, cpu);

		// Resume an idle thread for every unidled collaborator
		for (size_t i = 0; i < idleCPUs.size(); ++i) {
			assert(idleCPUs[i] != nullptr);
			ThreadManager::resumeIdle(idleCPUs[i]);
		}
	}
}


/*    IDLE CPUS    */

bool CPUManagerInterface::cpuBecomesIdle(CPU *cpu, bool inShutdown)
{
	assert(cpu != nullptr);

	const int index = cpu->getIndex();

	_idleCPUsLock.lock();

	if (!inShutdown) {
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
	}

	_idleCPUs[index] = true;
	++_numIdleCPUs;
	assert(_numIdleCPUs <= _cpus.size());

	Monitoring::cpuBecomesIdle(index);
	_idleCPUsLock.unlock();

	return true;
}

CPU *CPUManagerInterface::getIdleCPU(bool)
{
	std::lock_guard<SpinLock> guard(_idleCPUsLock);
	boost::dynamic_bitset<>::size_type idleCPU = _idleCPUs.find_first();
	if (idleCPU != boost::dynamic_bitset<>::npos) {
		_idleCPUs[idleCPU] = false;
		assert(_numIdleCPUs > 0);

		--_numIdleCPUs;
		Monitoring::cpuBecomesActive(idleCPU);
		return _cpus[idleCPU];
	} else {
		return nullptr;
	}
}

size_t CPUManagerInterface::getIdleCPUs(std::vector<CPU *> &idleCPUs, size_t numCPUs)
{
	assert(idleCPUs.size() >= numCPUs);

	size_t numObtainedCPUs = 0;

	std::lock_guard<SpinLock> guard(_idleCPUsLock);
	boost::dynamic_bitset<>::size_type idleCPU = _idleCPUs.find_first();
	while (numObtainedCPUs < numCPUs && idleCPU != boost::dynamic_bitset<>::npos) {
		// Signal that the CPU becomes active
		_idleCPUs[idleCPU] = false;
		Monitoring::cpuBecomesActive(idleCPU);

		// Place the CPU in the vector
		idleCPUs[numObtainedCPUs] = _cpus[idleCPU];
		++numObtainedCPUs;

		// Iterate to the next idle CPU
		idleCPU = _idleCPUs.find_next(idleCPU);
	}

	// Decrease the counter of idle CPUs by the obtained amount
	assert(_numIdleCPUs >= numObtainedCPUs);
	_numIdleCPUs -= numObtainedCPUs;

	return numObtainedCPUs;
}

CPU *CPUManagerInterface::getIdleNUMANodeCPU(size_t NUMANodeId)
{
	std::lock_guard<SpinLock> guard(_idleCPUsLock);
	boost::dynamic_bitset<> tmpIdleCPUs = _idleCPUs & _NUMANodeMask[NUMANodeId];
	boost::dynamic_bitset<>::size_type idleCPU = tmpIdleCPUs.find_first();
	if (idleCPU != boost::dynamic_bitset<>::npos) {
		_idleCPUs[idleCPU] = false;
		assert(_numIdleCPUs > 0);

		--_numIdleCPUs;
		Monitoring::cpuBecomesActive(idleCPU);
		return _cpus[idleCPU];
	} else {
		return nullptr;
	}
}

bool CPUManagerInterface::unidleCPU(CPU *cpu)
{
	assert(cpu != nullptr);

	const int index = cpu->getIndex();

	std::lock_guard<SpinLock> guard(_idleCPUsLock);
	if (_idleCPUs[index]) {
		_idleCPUs[index] = false;
		assert(_numIdleCPUs > 0);

		--_numIdleCPUs;
		Monitoring::cpuBecomesActive(index);
		return true;
	} else {
		return false;
	}
}

void CPUManagerInterface::getIdleCollaborators(std::vector<CPU *> &idleCPUs, ComputePlace *cpu)
{
	assert(cpu != nullptr);

	size_t numObtainedCollaborators = 0;

	std::lock_guard<SpinLock> guard(_idleCPUsLock);
	boost::dynamic_bitset<>::size_type idleCPU = _idleCPUs.find_first();
	while (idleCPU != boost::dynamic_bitset<>::npos) {
		assert(_cpus[idleCPU] != nullptr);

		if (((CPU *) cpu)->getGroupId() == _cpus[idleCPU]->getGroupId()) {
			// Signal that the CPU becomes active
			_idleCPUs[idleCPU] = false;
			Monitoring::cpuBecomesActive(idleCPU);
			++numObtainedCollaborators;

			// Place the CPU in the vector
			idleCPUs.push_back(_cpus[idleCPU]);
		}

		// Iterate to the next idle CPU
		idleCPU = _idleCPUs.find_next(idleCPU);
	}

	// Decrease the counter of idle CPUs by the obtained amount
	assert(_numIdleCPUs >= numObtainedCollaborators);
	_numIdleCPUs -= numObtainedCollaborators;
}
