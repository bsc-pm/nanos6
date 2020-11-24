/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef READY_QUEUE_MAP_HPP
#define READY_QUEUE_MAP_HPP

#include "memory/numa/NUMAManager.hpp"
#include "scheduling/ReadyQueue.hpp"
#include "support/Containers.hpp"
#include "tasks/Task.hpp"

// This kind of ready queue supports priorities
class ReadyQueueMap : public ReadyQueue {
	typedef Container::deque<Task *> ready_queue_t;
	typedef Container::map<Task::priority_t, ready_queue_t, std::greater<Task::priority_t>> ready_map_t;

	ready_map_t _readyMap;

	size_t _numReadyTasks;

public:
	ReadyQueueMap(SchedulingPolicy policy)
		: ReadyQueue(policy),
		_numReadyTasks(0)
	{
	}

	~ReadyQueueMap()
	{
		for (ready_map_t::iterator it = _readyMap.begin(); it != _readyMap.end(); it++) {
			assert(it->second.empty());
		}
		_readyMap.clear();
	}

	void addReadyTask(Task *task, bool unblocked)
	{
		Task::priority_t priority = task->getPriority();

		// Get ready queue for the given priority, if exists. If not, create it, and return it.
		ready_map_t::iterator it = (_readyMap.emplace(priority, ready_queue_t())).first;
		assert(it != _readyMap.end());

		if (unblocked || _policy == SchedulingPolicy::LIFO_POLICY) {
			it->second.push_front(task);
		} else {
			it->second.push_back(task);
		}

		++_numReadyTasks;
	}

	Task *getReadyTask(ComputePlace *)
	{
		if (_numReadyTasks == 0) {
			return nullptr;
		}

		ready_map_t::iterator it = _readyMap.begin();
		while (it != _readyMap.end()) {
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

		FatalErrorHandler::failIf(true, "There must be a ready task.");
		return nullptr;
	}

	inline size_t getNumReadyTasks() const
	{
		return _numReadyTasks;
	}

};


#endif // READY_QUEUE_MAP_HPP
