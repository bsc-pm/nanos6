/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef READY_QUEUE_MAP_HPP
#define READY_QUEUE_MAP_HPP

#include "dependencies/DataTrackingSupport.hpp"
#include "scheduling/ReadyQueue.hpp"
#include "support/Containers.hpp"
#include "tasks/Task.hpp"

// This kind of ready queue supports priorities
class ReadyQueueMap : public ReadyQueue {
	typedef Container::deque<Task *> ready_queue_t;
	typedef Container::map<Task::priority_t, ready_queue_t, std::greater<Task::priority_t>> ready_map_t;

	ready_map_t *_readyMaps;
	size_t *_numCurrentReadyTasksPerNUMANode;
	size_t _numReadyTasks;
	uint8_t _numMaps;
	// When tasks does not have a NUMA hint, we assign it in a round robin basis.
	uint8_t _roundRobinQueues;

public:
	ReadyQueueMap(SchedulingPolicy policy)
		: ReadyQueue(policy),
		_numReadyTasks(0),
		_roundRobinQueues(0)
	{
		_numMaps = DataTrackingSupport::isNUMASchedulingEnabled() ?
			HardwareInfo::getValidMemoryPlaceCount(nanos6_host_device) : 1;
		_readyMaps = (ready_map_t *) MemoryAllocator::alloc(_numMaps * sizeof(ready_map_t));
		_numCurrentReadyTasksPerNUMANode = (size_t *) MemoryAllocator::alloc(_numMaps * sizeof(size_t));

		for (uint8_t i = 0; i < _numMaps; i++) {
			new (&_readyMaps[i]) ready_map_t();
			_numCurrentReadyTasksPerNUMANode[i] = 0;
		}
	}

	~ReadyQueueMap()
	{
		for (uint8_t i = 0; i < _numMaps; i++) {
			for (ready_map_t::iterator it = _readyMaps[i].begin(); it != _readyMaps[i].end(); it++) {
				assert(it->second.empty());
			}
			_readyMaps[i].clear();
			_readyMaps[i].~ready_map_t();
		}
		MemoryAllocator::free(_readyMaps, _numMaps * sizeof(ready_map_t));
		MemoryAllocator::free(_numCurrentReadyTasksPerNUMANode, _numMaps * sizeof(size_t));
	}

	void addReadyTask(Task *task, bool unblocked)
	{
		Task::priority_t priority = task->getPriority();

		uint8_t NUMAid = 0;
		if (DataTrackingSupport::isNUMASchedulingEnabled()) {
			NUMAid = task->getNUMAhint();
			if (NUMAid == (uint8_t) -1) {
				NUMAid = _roundRobinQueues;
				_roundRobinQueues = (_roundRobinQueues + 1) % _numMaps;
			}
		}

		assert(NUMAid < _numMaps);

		// Get ready queue for the given priority, if exists. If not, create it, and return it.
		ready_map_t::iterator it = (_readyMaps[NUMAid].emplace(priority, ready_queue_t())).first;
		assert(it != _readyMaps[NUMAid].end());

		if (unblocked || _policy == SchedulingPolicy::LIFO_POLICY) {
			it->second.push_front(task);
		} else {
			it->second.push_back(task);
		}

		++_numReadyTasks;
		++_numCurrentReadyTasksPerNUMANode[NUMAid];
	}

	Task *getReadyTask(ComputePlace *computePlace)
	{
		if (_numReadyTasks == 0) {
			return nullptr;
		}

		uint8_t NUMAid = DataTrackingSupport::isNUMASchedulingEnabled() ? ((CPU *)computePlace)->getNumaNodeId() : 0;
		// 1. Try to get from my NUMA queue.
		if (!_readyMaps[NUMAid].empty()) {
			ready_map_t::iterator it = _readyMaps[NUMAid].begin();
			while (it != _readyMaps[NUMAid].end()) {
				if (it->second.empty()) {
					it++;
				} else {
					Task *result = it->second.front();
					assert(result != nullptr);

					it->second.pop_front();

					--_numReadyTasks;
					--_numCurrentReadyTasksPerNUMANode[NUMAid];

					return result;
				}
			}
		}

		// 2. Try to steal from other NUMA queues.
		// Stealing must consider distance and load balance
		if (DataTrackingSupport::isNUMAStealingEnabled()) {
			const std::vector<uint64_t> &distances = HardwareInfo::getNUMADistances();
			// Ideally, we want to steal from closer sockets with many tasks
			// We will use this score function: score = 100/distance + ready_tasks/5
			// The highest score, the better
			uint64_t score = 0;
			int16_t chosen = -1;
			for (uint8_t i = 0; i < _numMaps; i++) {
				if (i != NUMAid) {
					if (_numCurrentReadyTasksPerNUMANode[i] > 0) {
						uint64_t distance = distances[i*_numMaps+NUMAid];
						uint64_t loadFactor = _numCurrentReadyTasksPerNUMANode[i];
						if (distance < 15 && loadFactor > 20) {
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

			assert(chosen != -1 && _numCurrentReadyTasksPerNUMANode[chosen] > 0);
			ready_map_t::iterator it = _readyMaps[chosen].begin();
			while (it != _readyMaps[chosen].end()) {
				if (it->second.empty()) {
					it++;
				} else {
					Task *result = it->second.front();
					assert(result != nullptr);

					it->second.pop_front();

					--_numReadyTasks;
					--_numCurrentReadyTasksPerNUMANode[chosen];

					return result;
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


#endif // READY_QUEUE_MAP_HPP
