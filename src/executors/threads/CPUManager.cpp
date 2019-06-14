/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <boost/dynamic_bitset.hpp>

#include "WorkerThread.hpp"
#include "hardware/HardwareInfo.hpp"
#include "system/RuntimeInfo.hpp"

#include "CPU.hpp"
#include "CPUManager.hpp"
#include "ThreadManager.hpp"

#include <cassert>
#include <sstream>


std::vector<CPU *> CPUManager::_cpus;
size_t CPUManager::_totalCPUs;
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
	_totalCPUs = 0;
	
	cpu_set_t processCPUMask;
	int rc = sched_getaffinity(0, sizeof(cpu_set_t), &processCPUMask);
	FatalErrorHandler::handle(rc, " when retrieving the affinity of the process");
	
	// Get NUMA nodes
	_NUMANodeMask.resize(HardwareInfo::getMemoryPlaceCount(nanos6_device_t::nanos6_host_device));
	
	// Get CPU objects that can run a thread
	std::vector<ComputePlace *> const &cpus = ((HostInfo *) HardwareInfo::getDeviceInfo(nanos6_device_t::nanos6_host_device))->getComputePlaces();
	
	size_t maxSystemCPUId = 0;
	for (auto const *computePlace : cpus) {
		CPU const *cpu = (CPU const *) computePlace;
		
		if (cpu->_systemCPUId > maxSystemCPUId) {
			maxSystemCPUId = cpu->_systemCPUId;
		}
	}
	
	_cpus.resize(cpus.size());
	_systemToVirtualCPUId.resize(maxSystemCPUId+1);
	
	// Structures used for runtime information
	boost::dynamic_bitset<> processCPUBitset(cpus.size());
	std::vector<boost::dynamic_bitset<>> NUMANodeSystemMask(_NUMANodeMask.size());
	
	for (size_t i = 0; i < _NUMANodeMask.size(); ++i) {
		_NUMANodeMask[i].resize(cpus.size());
		NUMANodeSystemMask[i].resize(cpus.size());
	}
	
	for (size_t i = 0; i < cpus.size(); ++i) {
		CPU *cpu = (CPU *)cpus[i];
		
		size_t virtualCPUId;
		if (CPU_ISSET(cpu->_systemCPUId, &processCPUMask)) {
			virtualCPUId = _totalCPUs;
			cpu->_virtualCPUId = virtualCPUId;
			_cpus[virtualCPUId] = cpu;
			++_totalCPUs;
			_NUMANodeMask[cpu->_NUMANodeId][virtualCPUId] = true;
			
			processCPUBitset[i] = true;
			NUMANodeSystemMask[cpu->_NUMANodeId][cpu->_systemCPUId] = true;
		} else {
			virtualCPUId = (size_t) ~0UL;
			cpu->_virtualCPUId = virtualCPUId;
		}
		_systemToVirtualCPUId[cpu->_systemCPUId] = cpu->_virtualCPUId;
	}
	
	RuntimeInfo::addEntry("initial_cpu_list", "Initial CPU List", cpumanager_internals::maskToRegionList(processCPUBitset, cpus.size()));
	for (size_t i = 0; i < NUMANodeSystemMask.size(); ++i) {
		std::ostringstream oss, oss2;
		
		oss << "numa_node_" << i << "_cpu_list";
		oss2 << "NUMA Node " << i << " CPU List";
		std::string cpuRegionList = cpumanager_internals::maskToRegionList(NUMANodeSystemMask[i], cpus.size());
		
		RuntimeInfo::addEntry(oss.str(), oss2.str(), cpuRegionList);
	}
	
	// Set all CPUs as not idle
	_idleCPUs.resize(_cpus.size());
	_idleCPUs.reset();
}


void CPUManager::initialize()
{
	for (size_t systemCPUId = 0; systemCPUId < _cpus.size(); ++systemCPUId) {
		if (_cpus[systemCPUId] != nullptr) {
			CPU *cpu = _cpus[systemCPUId];
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
