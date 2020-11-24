/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include "UnsyncScheduler.hpp"
#include "dependencies/DataTrackingSupport.hpp"
#include "executors/threads/CPUManager.hpp"
#include "scheduling/ready-queues/ReadyQueueDeque.hpp"
#include "scheduling/ready-queues/ReadyQueueMap.hpp"

UnsyncScheduler::UnsyncScheduler(
	SchedulingPolicy policy,
	bool enablePriority,
	bool enableImmediateSuccessor
) :
	_deadlineTasks(nullptr),
	_roundRobinQueues(0),
	_enableImmediateSuccessor(enableImmediateSuccessor),
	_enablePriority(enablePriority)
{
	_numQueues = NUMAManager::getTrackingNodes();
	assert(_numQueues > 0);

	_queues = (ReadyQueue **) MemoryAllocator::alloc(_numQueues * sizeof(ReadyQueue *));
	assert(_queues != nullptr);

	for (uint64_t i = 0; i < _numQueues; i++) {
		if (NUMAManager::isValidNUMA(i)) {
			if (enablePriority) {
				_queues[i] = new ReadyQueueMap(policy);
			} else {
				_queues[i] = new ReadyQueueDeque(policy);
			}
		} else {
			_queues[i] = nullptr;
		}
	}

	if (enableImmediateSuccessor) {
		_immediateSuccessorTasks = immediate_successor_tasks_t(CPUManager::getTotalCPUs(), nullptr);
	}
}

UnsyncScheduler::~UnsyncScheduler()
{
	for (uint64_t i = 0; i < _numQueues; i++) {
		delete _queues[i];
	}

	if (_enablePriority) {
		MemoryAllocator::free(_queues, _numQueues * sizeof(ReadyQueue *));
	}
}

void UnsyncScheduler::regularAddReadyTask(Task *task, ReadyTaskHint hint)
{
	uint64_t NUMAid = task->getNUMAHint();
	//std::cout << "[Pre]NUMAid: " << NUMAid << std::endl;
	// In case there is no hint, use round robin to balance the load
	if (NUMAid == (uint64_t) -1) {
		do {
			NUMAid = _roundRobinQueues;
			_roundRobinQueues = (_roundRobinQueues + 1) % _numQueues;
		} while (_queues[NUMAid] == nullptr);
	}

	assert(NUMAid < _numQueues);

	assert(_queues[NUMAid] != nullptr);
	_queues[NUMAid]->addReadyTask(task, hint == UNBLOCKED_TASK_HINT);
}

Task *UnsyncScheduler::regularGetReadyTask(ComputePlace *computePlace)
{
	uint64_t NUMAid = 0;
	if (_numQueues > 1) {
		NUMAid =  ((CPU *)computePlace)->getNumaNodeId();
	}

	Task *result = nullptr;
	result = _queues[NUMAid]->getReadyTask(computePlace);

	if (result != nullptr)
		return result;

	if (_numQueues > 1) {
		// Try to steal.
		// Stealing must consider distance and load balance
		const std::vector<uint64_t> &distances = HardwareInfo::getNUMADistances();
		// Ideally, we want to steal from closer sockets with many tasks
		// We will use this score function: score = 100/distance + ready_tasks/5
		// The highest score, the better
		uint64_t score = 0;
		uint64_t chosen = (uint64_t) -1;
		for (uint64_t i = 0; i < _numQueues; i++) {
			if (i != NUMAid && _queues[i] != nullptr) {
				size_t numReadyTasks;
				numReadyTasks = _queues[i]->getNumReadyTasks();

				if (numReadyTasks > 0) {
					uint64_t distance = distances[i*_numQueues+NUMAid];
					uint64_t loadFactor = numReadyTasks;
					if (distance < DataTrackingSupport::getDistanceThreshold() &&
							loadFactor > DataTrackingSupport::getLoadThreshold())
					{
						chosen = i;
						break;
					}
					uint64_t tmpscore = 100/distance + loadFactor/5;
					if (tmpscore >= score) {
						score = tmpscore;
						chosen = i;
					}
				}
			}
		}

		if (chosen != (uint64_t) -1) {
			result = _queues[chosen]->getReadyTask(computePlace);
			assert(result != nullptr);

			return result;
		}
	}

	return result;
}
