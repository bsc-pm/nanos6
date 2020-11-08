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

	ready_queue_t _readyDeque;
public:
	ReadyQueueDeque(SchedulingPolicy policy) :
		ReadyQueue(policy)
	{
	}

	~ReadyQueueDeque()
	{
		assert(_readyDeque.empty());
	}

	void addReadyTask(Task *task, bool unblocked)
	{
		if (unblocked || _policy == SchedulingPolicy::LIFO_POLICY) {
			_readyDeque.push_front(task);
		} else {
			_readyDeque.push_back(task);
		}
	}

	Task *getReadyTask(ComputePlace *)
	{
		if (_readyDeque.empty()) {
			return nullptr;
		}

		Task *result = _readyDeque.front();
		assert(result != nullptr);

		_readyDeque.pop_front();
		return result;
	}

	inline size_t getNumReadyTasks() const
	{
		return _readyDeque.size();
	}

};


#endif // READY_QUEUE_DEQUE_HPP
