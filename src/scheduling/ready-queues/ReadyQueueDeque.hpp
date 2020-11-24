/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef READY_QUEUE_DEQUE_HPP
#define READY_QUEUE_DEQUE_HPP

#include "memory/numa/NUMAManager.hpp"
#include "scheduling/ReadyQueue.hpp"
#include "support/Containers.hpp"

class ReadyQueueDeque : public ReadyQueue {
	typedef Container::deque<Task *> ready_queue_t;

	ready_queue_t _readyDeque;
	size_t _numReadyTasks;

public:
	ReadyQueueDeque(SchedulingPolicy policy) :
		ReadyQueue(policy), _numReadyTasks(0)
	{
	}

	~ReadyQueueDeque()
	{
	}

	void addReadyTask(Task *task, bool unblocked)
	{
		if (unblocked || _policy == SchedulingPolicy::LIFO_POLICY) {
			_readyDeque.push_front(task);
		} else {
			_readyDeque.push_back(task);
		}

		++_numReadyTasks;
	}

	Task *getReadyTask(ComputePlace *)
	{
		if (_numReadyTasks == 0) {
			return nullptr;
		}

		Task *result = _readyDeque.front();
		assert(result != nullptr);

		_readyDeque.pop_front();

		--_numReadyTasks;

		return result;
	}

	inline size_t getNumReadyTasks() const
	{
		return _numReadyTasks;
	}

};


#endif // READY_QUEUE_DEQUE_HPP
