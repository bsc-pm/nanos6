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
	std::vector<ComputePlace *> computeNodes = HardwareInfo::getComputeNodes();

	for (size_t numa = 0; numa < NUMANodeCount; ++numa) {
		_readyTasks[numa] = 0;
		_cpuMask[numa].resize(computeNodes.size());

		for (ComputePlace *place : computeNodes) {
			if (((CPU *)place)->_NUMANodeId == numa) {
				_cpuMask[numa][((CPU *)place)->_systemCPUId] = true;
			}
		}
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
	size_t NUMANodeCount = HardwareInfo::getMemoryNodeCount();
	
	/* Get the least loaded NUMA node */
	int min_load = -1;
	int min_idx = -1;

	for (size_t numa = 0; numa < NUMANodeCount; ++numa) {
		if (_cpuMask[numa].any()) {
			if (min_load == -1 || _readyTasks[numa] < min_load) {
				min_load = _readyTasks[numa];
				min_idx = numa;
			}
		}
	}

	if (min_idx != -1) {
		_readyTasks[min_idx] += 1;
		return _NUMANodeScheduler[min_idx]->addReadyTask(task, hardwarePlace, hint);
	} else {
		return nullptr;
	}
}


void NUMAHierarchicalScheduler::taskGetsUnblocked(Task *unblockedTask, ComputePlace *hardwarePlace)
{
	/* This is not pretty. But this function is seldom called */
	CPU *cpu = unblockedTask->getThread()->getComputePlace();
	size_t numa_node = cpu->_NUMANodeId;

	assert(_cpuMask[numa_node][cpu->_systemCPUId]);

	_readyTasks[numa_node] += 1;
	_NUMANodeScheduler[numa_node]->taskGetsUnblocked(unblockedTask, hardwarePlace);
}


Task *NUMAHierarchicalScheduler::getReadyTask(ComputePlace *hardwarePlace, Task *currentTask)
{
	size_t numa_node = ((CPU *)hardwarePlace)->_NUMANodeId;
	Task *task = nullptr;

	assert(_cpuMask[numa_node][((CPU *)hardwarePlace)->_systemCPUId]);
	
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
		size_t NUMANodeCount = HardwareInfo::getMemoryNodeCount();
		ComputePlace *computePlace = nullptr;

		for (size_t numa = 0; numa < NUMANodeCount; ++numa) {
			if (_cpuMask[numa].any() && _readyTasks[numa] != 0) {
				computePlace = _NUMANodeScheduler[numa]->getIdleComputePlace(false);
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
	size_t numa_node = ((CPU *)hardwarePlace)->_NUMANodeId;

	_cpuMask[numa_node][((CPU *)hardwarePlace)->_systemCPUId] = false;
	_NUMANodeScheduler[numa_node]->disableComputePlace(hardwarePlace);
}

void NUMAHierarchicalScheduler::enableComputePlace(ComputePlace *hardwarePlace)
{
	size_t numa_node = ((CPU *)hardwarePlace)->_NUMANodeId;

	_cpuMask[numa_node][((CPU *)hardwarePlace)->_systemCPUId] = true;
	_NUMANodeScheduler[numa_node]->enableComputePlace(hardwarePlace);
}
