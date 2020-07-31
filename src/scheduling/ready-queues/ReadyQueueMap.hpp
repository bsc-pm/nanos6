/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef READY_QUEUE_MAP_HPP
#define READY_QUEUE_MAP_HPP

#include "scheduling/ReadyQueue.hpp"
#include "support/Containers.hpp"
#include "tasks/Task.hpp"

// This kind of ready queue supports priorities
class ReadyQueueMap : public ReadyQueue {
	typedef Container::deque<Task *> ready_queue_t;
	typedef Container::map<Task::priority_t, ready_queue_t, std::greater<Task::priority_t>> ready_map_t;

	ready_map_t *_readyMaps;
	size_t *_numReadyTasksPerNUMANode;
	uint8_t _numMaps;
	// When tasks does not have a NUMA hint, we assign it in a round robin basis.
	uint8_t _roundRobinQueues;

	size_t _numReadyTasks;
	size_t _enqueuedTasks;
public:
	ReadyQueueMap(SchedulingPolicy policy)
		: ReadyQueue(policy),
		_roundRobinQueues(0),
		_numReadyTasks(0)
	{
		_numMaps = DataTrackingSupport::isNUMATrackingEnabled() ?
			HardwareInfo::getValidMemoryPlaceCount(nanos6_host_device) : 1;
		_readyMaps = (ready_map_t *) MemoryAllocator::alloc(_numMaps * sizeof(ready_map_t));
		_numReadyTasksPerNUMANode = (size_t *) MemoryAllocator::alloc(_numMaps * sizeof(size_t));

		for (uint8_t i = 0; i < _numMaps; i++) {
			new (&_readyMaps[i]) ready_map_t();
			_numReadyTasksPerNUMANode[i] = 0;
		}
		_enqueuedTasks = 0;
	}

	~ReadyQueueMap()
	{
		for (uint8_t i = 0; i < _numMaps; i++) {
			for (ready_map_t::iterator it = _readyMaps[i].begin(); it != _readyMaps[i].end(); it++) {
				assert(it->second.empty());
			}
			_readyMaps[i].clear();
			_readyMaps[i].~ready_map_t();
			std::cout << "Tasks in NUMA node " << (int) i << ": " << _numReadyTasksPerNUMANode[i] << std::endl;
		}
		MemoryAllocator::free(_readyMaps, _numMaps * sizeof(ready_map_t));
		MemoryAllocator::free(_numReadyTasksPerNUMANode, _numMaps * sizeof(size_t));
		std::cout << "Tasks with no NUMA hint: " << (int) _roundRobinQueues << std::endl;
		std::cout << "Total number of enqueued tasks: " << _enqueuedTasks << std::endl;;
	}

	void addReadyTask(Task *task, bool unblocked)
	{
		Task::priority_t priority = task->getPriority();

		uint8_t NUMAid = 0;
		if (DataTrackingSupport::isNUMATrackingEnabled()) {
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
		++_numReadyTasksPerNUMANode[NUMAid];
		++_enqueuedTasks;
	}

	Task *getReadyTask(ComputePlace *computePlace)
	{
		if (_numReadyTasks == 0) {
			return nullptr;
		}

		uint8_t NUMAid = DataTrackingSupport::isNUMATrackingEnabled() ? ((CPU *)computePlace)->getNumaNodeId() : 0;
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
					return result;
				}
			}
		}

		// 2. Try to steal from other NUMA queues.
		if (DataTrackingSupport::isNUMAStealingEnabled()) {
			for (uint8_t i = 0; i < _numMaps; i++) {
				if (i != NUMAid) {
					if (!_readyMaps[i].empty()) {
						ready_map_t::iterator it = _readyMaps[i].begin();
						while (it != _readyMaps[i].end()) {
							if (it->second.empty()) {
								it++;
							} else {
								Task *result = it->second.front();
								assert(result != nullptr);

								it->second.pop_front();
								--_numReadyTasks;
								return result;
							}
						}
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


#endif // READY_QUEUE_MAP_HPP
