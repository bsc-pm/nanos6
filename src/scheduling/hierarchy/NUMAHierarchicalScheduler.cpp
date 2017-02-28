#include "HostHierarchicalScheduler.hpp"
#include "NUMAHierarchicalScheduler.hpp"

#include "../DefaultScheduler.hpp"
#include "../FIFOImmediateSuccessorWithPollingScheduler.hpp"
#include "../FIFOScheduler.hpp"
#include "../ImmediateSuccessorScheduler.hpp"
#include "../ImmediateSuccessorWithPollingScheduler.hpp"
#include "../PriorityScheduler.hpp"
#include "../Scheduler.hpp"
#include "../SchedulerInterface.hpp"

#include "executors/threads/CPUManager.hpp"
#include "hardware/HardwareInfo.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "executors/threads/WorkerThread.hpp"

#include <cassert>


NUMAHierarchicalScheduler::NUMAHierarchicalScheduler()
	: _NUMANodeScheduler(HardwareInfo::getMemoryNodeCount()),
	_readyTasks(HardwareInfo::getMemoryNodeCount())
{
	size_t NUMANodeCount = HardwareInfo::getMemoryNodeCount();

	for (size_t i = 0; i < NUMANodeCount; ++i) {
		_readyTasks[i] = 0;
	}

	EnvironmentVariable<std::string> schedulerName("NANOS6_SCHEDULER", "default");

	if (schedulerName.getValue() == "default") {
		for (size_t idx = 0; idx < NUMANodeCount; ++idx) {
			_NUMANodeScheduler[idx] = new DefaultScheduler();
		}
	} else if (schedulerName.getValue() == "fifo") {
		for (size_t idx = 0; idx < NUMANodeCount; ++idx) {
			_NUMANodeScheduler[idx] = new FIFOScheduler();
		}
	} else if (schedulerName.getValue() == "immediatesuccessor") {
		for (size_t idx = 0; idx < NUMANodeCount; ++idx) {
			_NUMANodeScheduler[idx] = new ImmediateSuccessorScheduler();
		}
	} else if (schedulerName.getValue() == "iswp") {
		for (size_t idx = 0; idx < NUMANodeCount; ++idx) {
			_NUMANodeScheduler[idx] = new ImmediateSuccessorWithPollingScheduler();
		}
	} else if (schedulerName.getValue() == "fifoiswp") {
		for (size_t idx = 0; idx < NUMANodeCount; ++idx) {
			_NUMANodeScheduler[idx] = new FIFOImmediateSuccessorWithPollingScheduler();
		}
	} else if (schedulerName.getValue() == "priority") {
		for (size_t idx = 0; idx < NUMANodeCount; ++idx) {
			_NUMANodeScheduler[idx] = new PriorityScheduler();
		}
	} else {
		std::cerr << "Warning: invalid scheduler name '" << schedulerName.getValue() << "', using default instead." << std::endl;
		for (size_t idx = 0; idx < NUMANodeCount; ++idx) {
			_NUMANodeScheduler[idx] = new DefaultScheduler();
		}
	}
	
}

NUMAHierarchicalScheduler::~NUMAHierarchicalScheduler()
{
	for (SchedulerInterface *sched : _NUMANodeScheduler) {
		delete sched;
	}
}


ComputePlace * NUMAHierarchicalScheduler::addReadyTask(Task *task, ComputePlace *hardwarePlace, ReadyTaskHint hint)
{
	/* Get the least loaded NUMA node */
	int min_load = _readyTasks[0];
	int min_idx = 0;

	for (size_t i = 1; i < _readyTasks.size(); ++i) {
		if (_readyTasks[i] < min_load) {
			min_load = _readyTasks[i];
			min_idx = i;
		}
	}

	_readyTasks[min_idx] += 1;
	return _NUMANodeScheduler[min_idx]->addReadyTask(task, hardwarePlace, hint);
}


void NUMAHierarchicalScheduler::taskGetsUnblocked(Task *unblockedTask, ComputePlace *hardwarePlace)
{
	/* This is not pretty. But this function is seldom called */
	size_t numa_node = unblockedTask->getThread()->getComputePlace()->_NUMANodeId;

	_readyTasks[numa_node] += 1;
	_NUMANodeScheduler[numa_node]->taskGetsUnblocked(unblockedTask, hardwarePlace);
}


Task *NUMAHierarchicalScheduler::getReadyTask(ComputePlace *hardwarePlace, Task *currentTask)
{
	size_t numa_node = ((CPU *)hardwarePlace)->_NUMANodeId;
	Task *task = nullptr;
	
	if (_readyTasks[numa_node] > 0) {
		task = _NUMANodeScheduler[numa_node]->getReadyTask(hardwarePlace, currentTask);

		if (task != nullptr) {
			_readyTasks[numa_node] -= 1;
		}
	} else {
		/* Get the most loaded NUMA node */
		int max_load = _readyTasks[numa_node];
		int max_idx = numa_node;

		for (size_t i = 0; i < _readyTasks.size(); ++i) {
			if (_readyTasks[i] > max_load) {
				max_load = _readyTasks[i];
				max_idx = i;
			}
		}
		
		task = _NUMANodeScheduler[max_idx]->getReadyTask(hardwarePlace, currentTask);
		if (task != nullptr) {
			_readyTasks[max_idx] -= 1;
		}
	}

	return task;
}


ComputePlace *NUMAHierarchicalScheduler::getIdleComputePlace(bool force)
{
	if (force) {
		return CPUManager::getIdleCPU();
	} else {
		ComputePlace *computePlace = nullptr;

		for (size_t i = 0; i < _readyTasks.size(); ++i) {
			if (_readyTasks[i] != 0) {
				computePlace = _NUMANodeScheduler[i]->getIdleComputePlace(false);
				if (computePlace != nullptr) {
					break;
				}
			}
		}

		return computePlace;
	}
}

