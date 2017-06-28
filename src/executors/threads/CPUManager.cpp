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
	static inline std::string maskToRangeList(boost::dynamic_bitset<> const &mask, std::vector<size_t> const &systemToVirtualCPUId)
	{
		std::ostringstream oss;
		
		int start = 0;
		int end = -1;
		bool first = true;
		for (size_t systemCPUId = 0; systemCPUId < mask.size()+1; systemCPUId++) {
			size_t virtualCPUId;
			
			if (systemCPUId < mask.size()) {
				assert(systemToVirtualCPUId.size() > systemCPUId);
				virtualCPUId = systemToVirtualCPUId[systemCPUId];
			} else {
				virtualCPUId = mask.size();
			}
			
			if ((virtualCPUId < mask.size()) && mask[virtualCPUId]) {
				if (end >= start) {
					// Valid range: extend
					end = systemCPUId;
				} else {
					// Invalid range: start
					start = systemCPUId;
					end = systemCPUId;
				}
			} else {
				if (end >= start) {
					// Valid range: emit and invalidate
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
					end = -1;
				} else {
					// Invalid range: do nothing
				}
			}
		}
		
		return oss.str();
	}
	
	
	static inline std::string maskToRangeList(cpu_set_t const &mask, size_t size)
	{
		std::ostringstream oss;
		
		int start = 0;
		int end = -1;
		bool first = true;
		for (size_t i = 0; i < size+1; i++) {
			if ((i < size) && CPU_ISSET(i, &mask)) {
				if (end >= start) {
					// Valid range: extend
					end = i;
				} else {
					// Invalid range: start
					start = i;
					end = i;
				}
			} else {
				if (end >= start) {
					// Valid range: emit and invalidate
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
					end = -1;
				} else {
					// Invalid range: do nothing
				}
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
	_NUMANodeMask.resize(HardwareInfo::getMemoryNodeCount());
	
	// Get CPU objects that can run a thread
	std::vector<ComputePlace *> const &cpus = HardwareInfo::getComputeNodes();
	_cpus.resize(cpus.size());
	_systemToVirtualCPUId.resize(cpus.size());
	
	for (size_t i = 0; i < _NUMANodeMask.size(); ++i) {
		_NUMANodeMask[i].resize(cpus.size());
	}
	
	for (size_t i = 0; i < cpus.size(); ++i) {
		CPU *cpu = (CPU *)cpus[i];
		
		_systemToVirtualCPUId[cpu->_systemCPUId] = cpu->_virtualCPUId;
		if (CPU_ISSET(cpu->_systemCPUId, &processCPUMask)) {
			_cpus[i] = cpu;
			++_totalCPUs;
			_NUMANodeMask[_cpus[i]->_NUMANodeId][i] = true;
		}
	}
	
	RuntimeInfo::addEntry("initial_cpu_list", "Initial CPU List", cpumanager_internals::maskToRangeList(processCPUMask, cpus.size()));
	for (size_t i = 0; i < _NUMANodeMask.size(); ++i) {
		std::ostringstream oss, oss2;
		
		oss << "numa_node_" << i << "_cpu_list";
		oss2 << "NUMA Node " << i << " CPU List";
		std::string cpuRangeList = cpumanager_internals::maskToRangeList(_NUMANodeMask[i], _systemToVirtualCPUId);
		
		RuntimeInfo::addEntry(oss.str(), oss2.str(), cpuRangeList);
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
				WorkerThread *initialThread = ThreadManager::getIdleThread(cpu, false);
				initialThread->resume(cpu, true);
			} else {
				// Already initialized?
			}
		}
	}

	_finishedCPUInitialization = true;
}
