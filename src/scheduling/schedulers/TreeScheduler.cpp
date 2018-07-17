/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#include <vector>

#include "executors/threads/CPUManager.hpp"
#include "hardware/HardwareInfo.hpp"
#include "tree-scheduler/LeafScheduler.hpp"
#include "tree-scheduler/NodeScheduler.hpp"
#include "TreeScheduler.hpp"

TreeScheduler::TreeScheduler() : _schedRRCounter(0)
{
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


TreeScheduler::~TreeScheduler() 
{
	delete _topScheduler;
}


ComputePlace *TreeScheduler::addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint, __attribute__((unused)) bool doGetIdle)
{
	assert(task != nullptr);
	FatalErrorHandler::failIf(task->isTaskloop(), "Task loop not supported yet"); // TODO
	
	if (computePlace != nullptr) {
		_CPUScheduler[((CPU *)computePlace)->_virtualCPUId]->addTask(task, true, hint);
	} else {
		// No computePlace means that this is the main task, or a task unblocked from outside
		LeafScheduler *sched = nullptr;
		while (sched == nullptr) {
			size_t pos = _schedRRCounter;
			while(!_schedRRCounter.compare_exchange_strong(pos, (pos + 1) % _CPUScheduler.size()));
			
			sched = _CPUScheduler[pos];
		}
		
		sched->addTask(task, false, hint);
	}
	
	return nullptr;
}


Task *TreeScheduler::getReadyTask(ComputePlace *computePlace, __attribute__((unused)) Task *currentTask, __attribute__((unused)) bool canMarkAsIdle)
{
	assert(computePlace != nullptr);
	
	return _CPUScheduler[((CPU *)computePlace)->_virtualCPUId]->getTask(false);
}


ComputePlace *TreeScheduler::getIdleComputePlace(__attribute__((unused)) bool force)
{
	return nullptr;
}


void TreeScheduler::disableComputePlace(ComputePlace *computePlace)
{
	assert(computePlace != nullptr);
	
	_CPUScheduler[((CPU *)computePlace)->_virtualCPUId]->disable();
}


void TreeScheduler::enableComputePlace(ComputePlace *computePlace)
{
	assert(computePlace != nullptr);
	
	_CPUScheduler[((CPU *)computePlace)->_virtualCPUId]->enable();
}

std::string TreeScheduler::getName() const
{
	return "tree";
}
