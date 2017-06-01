#include "NUMAHierarchicalScheduler.hpp"

#include "../SchedulerInterface.hpp"
#include "../SchedulerGenerator.hpp"

#include "hardware/HardwareInfo.hpp"
#include "executors/threads/TaskFinalization.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "system/RuntimeInfo.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskloopInfo.hpp"
#include "tasks/TaskloopManagerImplementation.hpp"

#include <InstrumentAddTask.hpp>

#include <cassert>
#include <sstream>


NUMAHierarchicalScheduler::NUMAHierarchicalScheduler()
	: _NUMANodeScheduler(HardwareInfo::getMemoryNodeCount()),
	_readyTasks(HardwareInfo::getMemoryNodeCount()),
	_enabledCPUs(HardwareInfo::getMemoryNodeCount())
{
	size_t NUMANodeCount = HardwareInfo::getMemoryNodeCount();
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
	if (task->isTaskloop()) {
		distributeTaskloopAmongNUMANodes((Taskloop *)task, computePlace, hint);
		return nullptr;
	} else {
		size_t NUMANodeCount = HardwareInfo::getMemoryNodeCount();
		
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


void NUMAHierarchicalScheduler::taskGetsUnblocked(Task *unblockedTask, ComputePlace *hardwarePlace)
{
	/* This is not pretty. But this function is seldom called */
	CPU *cpu = unblockedTask->getThread()->getComputePlace();
	size_t numa_node = cpu->_NUMANodeId;

	_readyTasks[numa_node] += 1;
	_NUMANodeScheduler[numa_node]->taskGetsUnblocked(unblockedTask, hardwarePlace);
}


Task *NUMAHierarchicalScheduler::getReadyTask(ComputePlace *computePlace, Task *currentTask, bool canMarkAsIdle)
{
	size_t numa_node = ((CPU *)computePlace)->_NUMANodeId;
	Task *task = nullptr;
	
	task = _NUMANodeScheduler[numa_node]->getReadyTask(computePlace, currentTask, false);
	
	if (task == nullptr) {
		for (size_t i = 0; i < _readyTasks.size(); ++i) {
			task = _NUMANodeScheduler[i]->getReadyTask(computePlace, currentTask, false);
			if (task != nullptr) {
				break;
			}
		}
	}
	
// FIXME
#if 0
	if (_readyTasks[numa_node] > 0) {
		task = _NUMANodeScheduler[numa_node]->getReadyTask(computePlace, currentTask, false);

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
		
		task = _NUMANodeScheduler[max_idx]->getReadyTask(computePlace, currentTask, false);
		if (task != nullptr) {
			_readyTasks[max_idx] -= 1;
		}
	}
#endif

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
		size_t NUMANodeCount = HardwareInfo::getMemoryNodeCount();
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

bool NUMAHierarchicalScheduler::requestPolling(ComputePlace *computePlace, polling_slot_t *pollingSlot)
{
	size_t NUMANode = ((CPU *)computePlace)->_NUMANodeId;
	return _NUMANodeScheduler[NUMANode]->requestPolling(computePlace, pollingSlot);
}

bool NUMAHierarchicalScheduler::releasePolling(ComputePlace *computePlace, polling_slot_t *pollingSlot)
{
	size_t NUMANode = ((CPU *)computePlace)->_NUMANodeId;
	return _NUMANodeScheduler[NUMANode]->releasePolling(computePlace, pollingSlot);
}

std::string NUMAHierarchicalScheduler::getName() const
{
	return "numa-hierarchical";
}

size_t NUMAHierarchicalScheduler::getAvailableNUMANodeCount()
{
	size_t NUMANodeCount = HardwareInfo::getMemoryNodeCount();
	size_t availableNUMANodeCount = 0;
	
	for (size_t numa = 0; numa < NUMANodeCount; ++numa) {
		if (_enabledCPUs[numa] > 0) {
			++availableNUMANodeCount;
		}
	}
	
	return availableNUMANodeCount;
}

void NUMAHierarchicalScheduler::distributeTaskloopAmongNUMANodes(Taskloop *taskloop, ComputePlace *computePlace, ReadyTaskHint hint)
{
	assert(taskloop != nullptr);
	assert(computePlace != nullptr);
	
	size_t NUMANodeCount = HardwareInfo::getMemoryNodeCount();
	size_t availableNUMANodeCount = getAvailableNUMANodeCount();
	assert(NUMANodeCount > 0);
	assert(availableNUMANodeCount > 0);
	assert(availableNUMANodeCount <= NUMANodeCount);
	
	// Get the information of the complete taskloop bounds
	const TaskloopInfo &taskloopInfo = taskloop->getTaskloopInfo();
	const nanos_taskloop_bounds &taskloopBounds = taskloopInfo._bounds;
	
	size_t originalLowerBound = taskloopBounds.lower_bound;
	size_t originalUpperBound = taskloopBounds.upper_bound;
	size_t grainSize = taskloopBounds.grain_size;
	size_t step = taskloopBounds.step;
	
	nanos_taskloop_bounds partialBounds;
	partialBounds.grain_size = grainSize;
	partialBounds.step = step;
	
	// Compute the actual number of iterations
	size_t totalIts = TaskloopBounds::getIterationCount(taskloopBounds);
	size_t itsPerPartition = totalIts / availableNUMANodeCount;
	size_t missalignment = itsPerPartition % grainSize;
	size_t extraItsPerPartition = (missalignment) ? grainSize - missalignment : 0;
	itsPerPartition += extraItsPerPartition;
	
	// Start the partition from the original lower bound
	size_t lowerBound = originalLowerBound;
	
	size_t numa = 0, partition = 0;
	while (partition < NUMANodeCount && numa < NUMANodeCount) {
		if (_enabledCPUs[numa] > 0) {
			// Compute the upper bound for this partition
			size_t upperBound = (partition < availableNUMANodeCount - 1) ?
				lowerBound + itsPerPartition * step : originalUpperBound;
			
			// Set taskloop bounds
			partialBounds.lower_bound = lowerBound;
			partialBounds.upper_bound = upperBound;
			
			// Create a partition taskloop for this NUMA node
			Taskloop *partitionTaskloop = TaskloopManager::createPartitionTaskloop(taskloop, partialBounds);
			assert(partitionTaskloop != nullptr);
			
			// Update the lower bound for the next partition
			lowerBound = upperBound;
			
			// Send the work to the NUMA Node
			_NUMANodeScheduler[numa]->addReadyTask(partitionTaskloop, computePlace, hint);
			++partition;
		}
		
		++numa;
	}
	
	if (taskloop->markAsFinished()) {
		TaskFinalization::disposeOrUnblockTask(taskloop, computePlace);
	}
}
