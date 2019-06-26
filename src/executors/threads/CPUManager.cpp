/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
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


std::vector<CPU *> CPUManager::_cpus;
std::atomic<bool> CPUManager::_finishedCPUInitialization;
SpinLock CPUManager::_idleCPUsLock;
boost::dynamic_bitset<> CPUManager::_idleCPUs;
std::vector<boost::dynamic_bitset<>> CPUManager::_NUMANodeMask;
std::vector<size_t> CPUManager::_systemToVirtualCPUId;


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
