/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#include <vector>

#include "hardware/HardwareInfo.hpp"
#include "LeafScheduler.hpp"
#include "NodeScheduler.hpp"
#include "Scheduler.hpp"

std::vector<LeafScheduler *> Scheduler::_CPUScheduler;
NodeScheduler *Scheduler::_topScheduler;

void Scheduler::initialize()
{
	_topScheduler = new NodeScheduler();
	
	size_t NUMANodeCount = HardwareInfo::getMemoryNodeCount();
	const std::vector<ComputePlace *> &computePlaces = HardwareInfo::getComputeNodes();
	
	_CPUScheduler.resize(computePlaces.size());
	
	if (NUMANodeCount > 1) {
		std::vector<NodeScheduler *> NUMAScheduler(NUMANodeCount, nullptr);
		for (size_t i = 0; i < NUMANodeCount; ++i) {
			NUMAScheduler[i] = new NodeScheduler(_topScheduler);
		}
		
		for (ComputePlace *computePlace : computePlaces) {
			CPU *cpu = (CPU *)computePlace;
			_CPUScheduler[cpu->_virtualCPUId] = new LeafScheduler(cpu, NUMAScheduler[cpu->_NUMANodeId]);
		}
	} else {
		for (ComputePlace *computePlace : computePlaces) {
			CPU *cpu = (CPU *)computePlace;
			_CPUScheduler[cpu->_virtualCPUId] = new LeafScheduler(cpu, _topScheduler);
		}
	}
}

void Scheduler::shutdown() 
{
	delete _topScheduler;
}

#include "instrument/support/InstrumentThreadLocalDataSupport.hpp"
#include "instrument/support/InstrumentThreadLocalDataSupportImplementation.hpp"
