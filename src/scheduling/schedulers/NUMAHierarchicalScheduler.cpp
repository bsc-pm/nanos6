/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "NUMAHierarchicalScheduler.hpp"

#include "../SchedulerInterface.hpp"
#include "../SchedulerGenerator.hpp"

#include "hardware/HardwareInfo.hpp"
#include "executors/threads/TaskFinalization.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "system/RuntimeInfo.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"

#include <InstrumentAddTask.hpp>

#include <cassert>
#include <sstream>
#include <vector>


NUMAHierarchicalScheduler::NUMAHierarchicalScheduler()
	: _NUMANodeScheduler(HardwareInfo::getMemoryPlaceCount(nanos6_device_t::nanos6_host_device)),
	_readyTasks(HardwareInfo::getMemoryPlaceCount(nanos6_device_t::nanos6_host_device)),
	_enabledCPUs(HardwareInfo::getMemoryPlaceCount(nanos6_device_t::nanos6_host_device))
{
	size_t NUMANodeCount = HardwareInfo::getMemoryPlaceCount(nanos6_device_t::nanos6_host_device);
	std::vector<CPU *> const &cpus = CPUManager::getCPUListReference();

	for (CPU *cpu : cpus) {
		if (cpu != nullptr) {
			_enabledCPUs[cpu->_NUMANodeId] += 1;
		}
	}

	for (size_t idx = 0; idx < NUMANodeCount; ++idx) {
		_NUMANodeScheduler[idx] = SchedulerGenerator::createNUMANodeScheduler(idx);
		
		std::ostringstream oss, oss2;
		oss << "numa-node-" << idx << "-scheduler";
		oss2 << "NUMA Node " << idx << " Scheduler";
		RuntimeInfo::addEntry(oss.str(), oss2.str(), _NUMANodeScheduler[idx]->getName());
	}
}

NUMAHierarchicalScheduler::~NUMAHierarchicalScheduler()
{
	for (SchedulerInterface *sched : _NUMANodeScheduler) {
		delete sched;
	}
}


ComputePlace * NUMAHierarchicalScheduler::addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint, bool doGetIdle)
{
	assert(task != nullptr);
	
	FatalErrorHandler::failIf(task->getDeviceType() != nanos6_device_t::nanos6_host_device, "Device tasks not supported by this scheduler");	
	FatalErrorHandler::failIf(task->isTaskloop(), "Task loop not supported by this scheduler");
	
	if (hint == UNBLOCKED_TASK_HINT) {
		CPU *cpu = task->getThread()->getComputePlace();
		size_t numa_node = cpu->_NUMANodeId;

		_readyTasks[numa_node] += 1;
		_NUMANodeScheduler[numa_node]->addReadyTask(task, computePlace, hint, doGetIdle);
		
		return nullptr;
	} else {
		size_t NUMANodeCount = HardwareInfo::getMemoryPlaceCount(nanos6_device_t::nanos6_host_device);
		
		/* Get the least loaded NUMA node */
		int min_load = -1;
		int min_idx = -1;
		
		for (size_t numa = 0; numa < NUMANodeCount; ++numa) {
			if (_enabledCPUs[numa] > 0) {
				if (min_load == -1 || _readyTasks[numa] < min_load) {
					min_load = _readyTasks[numa];
					min_idx = numa;
				}
			}
		}
		
		assert(min_idx != -1);
		
		_readyTasks[min_idx] += 1;
		_NUMANodeScheduler[min_idx]->addReadyTask(task, computePlace, hint, false);
		if (doGetIdle) {
			ComputePlace *cp;
			cp = CPUManager::getIdleNUMANodeCPU(min_idx);
			if (cp == nullptr) {
				// If this NUMA node does not have any idle CPUs, get any other idle CPU
				cp = CPUManager::getIdleCPU();
			}
		
			return cp;
		} else {
			return nullptr;
		}
	}
}


Task *NUMAHierarchicalScheduler::getReadyTask(ComputePlace *computePlace, Task *currentTask, bool canMarkAsIdle, bool doWait)
{
	if (computePlace->getType() != nanos6_device_t::nanos6_host_device) {
		return nullptr;
	}
	
	size_t numa_node = ((CPU *)computePlace)->_NUMANodeId;
	Task *task = nullptr;
	
	if (_readyTasks[numa_node] > 0) {
		task = _NUMANodeScheduler[numa_node]->getReadyTask(computePlace, currentTask, false, doWait);

		if (task != nullptr) {
			_readyTasks[numa_node] -= 1;
		}
	}

	if (task == nullptr) {
		/* Get the most loaded NUMA node */
		int max_load = _readyTasks[numa_node];
		int max_idx = numa_node;

		for (size_t i = 0; i < _readyTasks.size(); ++i) {
			if (_readyTasks[i] > max_load) {
				max_load = _readyTasks[i];
				max_idx = i;
			}
		}
		
		task = _NUMANodeScheduler[max_idx]->getReadyTask(computePlace, currentTask, false, doWait);
		if (task != nullptr) {
			_readyTasks[max_idx] -= 1;
		}
	}
	
	if (canMarkAsIdle && task == nullptr) {
		CPUManager::cpuBecomesIdle((CPU *) computePlace);
	}
	
	return task;
}


ComputePlace *NUMAHierarchicalScheduler::getIdleComputePlace(bool force)
{
	if (force) {
		return CPUManager::getIdleCPU();
	} else {
		size_t NUMANodeCount = HardwareInfo::getMemoryPlaceCount(nanos6_device_t::nanos6_host_device);
		ComputePlace *computePlace = nullptr;

		for (size_t numa = 0; numa < NUMANodeCount; ++numa) {
			if (_enabledCPUs[numa] > 0 && _readyTasks[numa] > 0) {
				computePlace = CPUManager::getIdleNUMANodeCPU(numa);
				if (computePlace != nullptr) {
					break;
				}
			}
		}

		return computePlace;
	}
}

void NUMAHierarchicalScheduler::disableComputePlace(ComputePlace *hardwarePlace)
{
	size_t NUMANode = ((CPU *)hardwarePlace)->_NUMANodeId;

	assert(_enabledCPUs[NUMANode] > 0);
	
	// TODO: do something if a NUMA node has pending tasks but is disabled.
	// This is not an issue now because of task stealing
	
	_enabledCPUs[NUMANode] -= 1;

	_NUMANodeScheduler[NUMANode]->disableComputePlace(hardwarePlace);
}

void NUMAHierarchicalScheduler::enableComputePlace(ComputePlace *hardwarePlace)
{
	size_t NUMANode = ((CPU *)hardwarePlace)->_NUMANodeId;

	_enabledCPUs[NUMANode] += 1;
	_NUMANodeScheduler[NUMANode]->enableComputePlace(hardwarePlace);
}

bool NUMAHierarchicalScheduler::requestPolling(ComputePlace *computePlace, polling_slot_t *pollingSlot, bool canMarkAsIdle)
{
	size_t NUMANode = ((CPU *)computePlace)->_NUMANodeId;
	return _NUMANodeScheduler[NUMANode]->requestPolling(computePlace, pollingSlot, canMarkAsIdle);
}

bool NUMAHierarchicalScheduler::releasePolling(ComputePlace *computePlace, polling_slot_t *pollingSlot, bool canMarkAsIdle)
{
	size_t NUMANode = ((CPU *)computePlace)->_NUMANodeId;
	return _NUMANodeScheduler[NUMANode]->releasePolling(computePlace, pollingSlot, canMarkAsIdle);
}

std::string NUMAHierarchicalScheduler::getName() const
{
	return "numa-hierarchical";
}

size_t NUMAHierarchicalScheduler::getAvailableNUMANodeCount()
{
	size_t NUMANodeCount = HardwareInfo::getMemoryPlaceCount(nanos6_device_t::nanos6_host_device);
	size_t availableNUMANodeCount = 0;
	
	for (size_t numa = 0; numa < NUMANodeCount; ++numa) {
		if (_enabledCPUs[numa] > 0) {
			++availableNUMANodeCount;
		}
	}
	
	return availableNUMANodeCount;
}
