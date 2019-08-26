/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#include <boost/dynamic_bitset.hpp>
#include <cassert>
#include <sched.h>
#include <sstream>

#include "CPU.hpp"
#include "CPUManager.hpp"
#include "ThreadManager.hpp"
#include "WorkerThread.hpp"
#include "hardware/HardwareInfo.hpp"
#include "system/RuntimeInfo.hpp"

#include <Monitoring.hpp>


std::vector<CPU *> CPUManager::_cpus;
std::atomic<bool> CPUManager::_finishedCPUInitialization;
SpinLock CPUManager::_idleCPUsLock;
boost::dynamic_bitset<> CPUManager::_idleCPUs;
std::vector<boost::dynamic_bitset<>> CPUManager::_NUMANodeMask;
std::vector<size_t> CPUManager::_systemToVirtualCPUId;
EnvironmentVariable<size_t> CPUManager::_taskforGroups("NANOS6_TASKFOR_GROUPS", 1);


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

size_t CPUManager::getNumCPUsPerTaskforGroup()
{
	return HardwareInfo::getComputePlaceCount(nanos6_host_device) / _taskforGroups;
}

void CPUManager::preinitialize()
{
	_finishedCPUInitialization = false;
	
	cpu_set_t processCPUMask;
	int rc = sched_getaffinity(0, sizeof(cpu_set_t), &processCPUMask);
	FatalErrorHandler::handle(rc, " when retrieving the affinity of the process");
	
	// Get NUMA nodes
	const size_t numNUMANodes = HardwareInfo::getMemoryPlaceCount(nanos6_device_t::nanos6_host_device);
	_NUMANodeMask.resize(numNUMANodes);
	
	// Get CPU objects that can run a thread
	std::vector<ComputePlace *> const &cpus = ((HostInfo *) HardwareInfo::getDeviceInfo(nanos6_device_t::nanos6_host_device))->getComputePlaces();
	if (cpus.size() < _taskforGroups) {
		FatalErrorHandler::warnIf(1, "You requested more groups than CPUs in the system. We are going to use ", cpus.size(), " groups of 1 CPU each.");
		_taskforGroups.setValue(cpus.size());
	}
	if (_taskforGroups == 0 || cpus.size() % _taskforGroups != 0) {
		size_t closestGroups = getClosestGroupNumber(cpus.size(), _taskforGroups);
		FatalErrorHandler::warnIf(_taskforGroups == 0, "You requested 0 groups, but 0 is not a valid number of groups. We are going to use ",
				closestGroups, " of ", cpus.size() / closestGroups, " CPUs each.");
		FatalErrorHandler::warnIf(_taskforGroups != 0 && cpus.size() % _taskforGroups != 0, "You requested ", _taskforGroups,
				" groups, but the number of CPUs is not divisible by the number of groups. We are going to use ",
				closestGroups, " of ", cpus.size() / closestGroups, " CPUs each.");
		_taskforGroups.setValue(closestGroups);
	}
	assert(_taskforGroups <= cpus.size() && cpus.size() % _taskforGroups == 0);
	
	size_t maxSystemCPUId = 0;
	for (size_t i = 0; i < cpus.size(); ++i) {
		const CPU *cpu = (const CPU *)cpus[i];
		
		if (cpu->getSystemCPUId() > maxSystemCPUId) {
			maxSystemCPUId = cpu->getSystemCPUId();
		}
	}
	
	const size_t numSystemCPUs = maxSystemCPUId + 1;
	const size_t numAvailableCPUs = CPU_COUNT(&processCPUMask);
	_cpus.resize(numAvailableCPUs);
	_systemToVirtualCPUId.resize(numSystemCPUs);
	
	for (size_t i = 0; i < numNUMANodes; ++i) {
		_NUMANodeMask[i].resize(numAvailableCPUs);
	}
	
	size_t virtualCPUId = 0;
	for (size_t i = 0; i < cpus.size(); ++i) {
		CPU *cpu = (CPU *)cpus[i];
		
		if (CPU_ISSET(cpu->getSystemCPUId(), &processCPUMask)) {
			// We need the hwloc logical_index to compute the groupId. However, that index is overwritten, so this is the last point where we still have
			// the hwloc logical_index, so we compute the groupId here and set it as member of CPU.
			size_t groupId = cpu->getIndex() / getNumCPUsPerTaskforGroup();
			assert(groupId <= cpus.size());
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
	
	CPUManager::reportInformation(numSystemCPUs, numNUMANodes);
	
	// Set all CPUs as not idle
	_idleCPUs.resize(numAvailableCPUs);
	_idleCPUs.reset();
}

void CPUManager::initialize()
{
	for (size_t virtualCPUId = 0; virtualCPUId < _cpus.size(); ++virtualCPUId) {
		if (_cpus[virtualCPUId] != nullptr) {
			CPU *cpu = _cpus[virtualCPUId];
			assert(cpu != nullptr);
			
			// Inform monitoring that the task becomes active by default
			Monitoring::cpuBecomesActive(cpu->getIndex());
			
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

void CPUManager::reportInformation(size_t numSystemCPUs, size_t numNUMANodes)
{
	boost::dynamic_bitset<> processCPUMask(numSystemCPUs);
	
	std::vector<boost::dynamic_bitset<>> NUMANodeSystemMask(numNUMANodes);
	for (size_t i = 0; i < numNUMANodes; ++i) {
		NUMANodeSystemMask[i].resize(numSystemCPUs);
	}
	
	for (CPU *cpu : _cpus) {
		size_t systemCPUId = cpu->getSystemCPUId();
		processCPUMask[systemCPUId] = true;
		NUMANodeSystemMask[cpu->getNumaNodeId()][systemCPUId] = true;
	}
	
	RuntimeInfo::addEntry("initial_cpu_list", "Initial CPU List", cpumanager_internals::maskToRegionList(processCPUMask, numSystemCPUs));
	for (size_t i = 0; i < numNUMANodes; ++i) {
		std::ostringstream oss, oss2;
		
		oss << "numa_node_" << i << "_cpu_list";
		oss2 << "NUMA Node " << i << " CPU List";
		std::string cpuRegionList = cpumanager_internals::maskToRegionList(NUMANodeSystemMask[i], numSystemCPUs);
		
		RuntimeInfo::addEntry(oss.str(), oss2.str(), cpuRegionList);
	}
}

void CPUManager::cpuBecomesIdle(CPU *cpu)
{
	const int index = cpu->getIndex();
	std::lock_guard<SpinLock> guard(_idleCPUsLock);
	_idleCPUs[index] = true;
	Monitoring::cpuBecomesIdle(index);
}

CPU *CPUManager::getIdleCPU()
{
	std::lock_guard<SpinLock> guard(_idleCPUsLock);
	boost::dynamic_bitset<>::size_type idleCPU = _idleCPUs.find_first();
	if (idleCPU != boost::dynamic_bitset<>::npos) {
		_idleCPUs[idleCPU] = false;
		Monitoring::cpuBecomesActive(idleCPU);
		return _cpus[idleCPU];
	} else {
		return nullptr;
	}
}

void CPUManager::getIdleCPUs(std::vector<CPU *> &idleCPUs)
{
	assert(idleCPUs.empty());
	
	std::lock_guard<SpinLock> guard(_idleCPUsLock);
	boost::dynamic_bitset<>::size_type idleCPU = _idleCPUs.find_first();
	while (idleCPU != boost::dynamic_bitset<>::npos) {
		_idleCPUs[idleCPU] = false;
		Monitoring::cpuBecomesActive(idleCPU);
		idleCPUs.push_back(_cpus[idleCPU]);
		idleCPU = _idleCPUs.find_next(idleCPU);
	}
}

CPU *CPUManager::getIdleNUMANodeCPU(size_t NUMANodeId)
{
	std::lock_guard<SpinLock> guard(_idleCPUsLock);
	boost::dynamic_bitset<> tmpIdleCPUs = _idleCPUs & _NUMANodeMask[NUMANodeId];
	boost::dynamic_bitset<>::size_type idleCPU = tmpIdleCPUs.find_first();
	if (idleCPU != boost::dynamic_bitset<>::npos) {
		_idleCPUs[idleCPU] = false;
		Monitoring::cpuBecomesActive(idleCPU);
		return _cpus[idleCPU];
	} else {
		return nullptr;
	}
}

bool CPUManager::unidleCPU(CPU *cpu)
{
	assert(cpu != nullptr);
	const int index = cpu->getIndex();
	
	std::lock_guard<SpinLock> guard(_idleCPUsLock);
	if (_idleCPUs[index]) {
		_idleCPUs[index] = false;
		Monitoring::cpuBecomesActive(index);
		return true;
	} else {
		return false;
	}
}
