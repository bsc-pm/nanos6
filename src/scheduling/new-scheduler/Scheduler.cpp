/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#include <vector>

#include "executors/threads/CPUManager.hpp"
#include "hardware/HardwareInfo.hpp"
#include "LeafScheduler.hpp"
#include "NodeScheduler.hpp"
#include "Scheduler.hpp"

std::vector<LeafScheduler *> Scheduler::_CPUScheduler;
NodeScheduler *Scheduler::_topScheduler;
std::atomic<size_t> Scheduler::_schedRRCounter;

void Scheduler::initialize()
{
	_schedRRCounter = 0;

	_topScheduler = new NodeScheduler();
	
	size_t NUMANodeCount = HardwareInfo::getMemoryNodeCount();
	std::vector<CPU *> const &cpus = CPUManager::getCPUListReference();
	
	_CPUScheduler.resize(cpus.size());
	
	if (NUMANodeCount > 1) {
		std::vector<NodeScheduler *> NUMAScheduler(NUMANodeCount, nullptr);
		for (size_t i = 0; i < NUMANodeCount; ++i) {
			NUMAScheduler[i] = new NodeScheduler(_topScheduler);
		}
		
		for (CPU *cpu : cpus) {
			if (cpu != nullptr) {
				_CPUScheduler[cpu->_virtualCPUId] = new LeafScheduler(cpu, NUMAScheduler[cpu->_NUMANodeId]);
			}
		}
	} else {
		for (CPU *cpu : cpus) {
			if (cpu != nullptr) {
				_CPUScheduler[cpu->_virtualCPUId] = new LeafScheduler(cpu, _topScheduler);
			}
		}
	}
}

void Scheduler::shutdown() 
{
	delete _topScheduler;
}

#include "instrument/support/InstrumentThreadLocalDataSupport.hpp"
#include "instrument/support/InstrumentThreadLocalDataSupportImplementation.hpp"
