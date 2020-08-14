/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef READY_QUEUE_DEQUE_HPP
#define READY_QUEUE_DEQUE_HPP

#include "scheduling/ReadyQueue.hpp"
#include "support/Containers.hpp"

class ReadyQueueDeque : public ReadyQueue {
	typedef Container::deque<Task *> ready_queue_t;

	ready_queue_t *_readyDeques;
	uint8_t _numQueues;
	// When tasks does not have a NUMA hint, we assign it in a round robin basis.
	uint8_t _roundRobinQueues;

	size_t _numReadyTasks;
public:
	ReadyQueueDeque(SchedulingPolicy policy)
		: ReadyQueue(policy), _roundRobinQueues(0), _numReadyTasks(0)
	{
		_numQueues = DataTrackingSupport::isNUMASchedulingEnabled() ?
			HardwareInfo::getValidMemoryPlaceCount(nanos6_host_device) : 1;
		_readyDeques = (ready_queue_t *) MemoryAllocator::alloc(_numQueues * sizeof(ready_queue_t));

		for (uint8_t i = 0; i < _numQueues; i++) {
			new (&_readyDeques[i]) ready_queue_t();
		}
	}

	~ReadyQueueDeque()
	{
		for (uint8_t i = 0; i < _numQueues; i++) {
			assert(_readyDeques[i].empty());
			_readyDeques[i].~ready_queue_t();
		}
		MemoryAllocator::free(_readyDeques, _numQueues * sizeof(ready_queue_t));
	}

	void addReadyTask(Task *task, bool unblocked)
	{
		uint8_t NUMAid = 0;
		if (DataTrackingSupport::isNUMASchedulingEnabled()) {
			NUMAid = task->getNUMAhint();
			if (NUMAid == (uint8_t) -1) {
				NUMAid = _roundRobinQueues;
				_roundRobinQueues = (_roundRobinQueues + 1) % _numQueues;
			}
		}

		assert(NUMAid < _numQueues);

		if (unblocked || _policy == SchedulingPolicy::LIFO_POLICY) {
			_readyDeques[NUMAid].push_front(task);
		} else {
			_readyDeques[NUMAid].push_back(task);
		}

		++_numReadyTasks;
	}

	Task *getReadyTask(ComputePlace *computePlace)
	{
		if (_numReadyTasks == 0) {
			return nullptr;
		}

		uint8_t NUMAid = DataTrackingSupport::isNUMASchedulingEnabled() ? ((CPU *)computePlace)->getNumaNodeId() : 0;
		// 1. Try to get from my NUMA queue.
		if (!_readyDeques[NUMAid].empty()) {
			Task *result = _readyDeques[NUMAid].front();
			assert(result != nullptr);

			_readyDeques[NUMAid].pop_front();

			--_numReadyTasks;

			return result;
		}

		// 2. Try to steal from other NUMA queues.
		if (DataTrackingSupport::isNUMAStealingEnabled()) {
			for (uint8_t i = 0; i < _numQueues; i++) {
				if (i != NUMAid) {
					if (!_readyDeques[i].empty()) {
						Task *result = _readyDeques[i].front();
						assert(result != nullptr);

						_readyDeques[i].pop_front();

						--_numReadyTasks;

						return result;
					}
				}
			}
		}

		FatalErrorHandler::failIf(DataTrackingSupport::isNUMAStealingEnabled(), "There must be a ready task.");
		return nullptr;
	}

	inline size_t getNumReadyTasks() const
	{
		return _numReadyTasks;
	}

};


#endif // READY_QUEUE_DEQUE_HPP
