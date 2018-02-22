/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#include "FIFOQueue.hpp"

size_t FIFOQueue::addTask(Task *task, __attribute__((unused)) SchedulerInterface::ReadyTaskHint hint)
{
	std::lock_guard<SpinLock> guard(_lock);
	
	_queue.push_back(task);
	
	return _queue.size();
}

size_t FIFOQueue::addTaskBatch(const std::vector<Task *> &taskBatch)
{
	std::lock_guard<SpinLock> guard(_lock);
	
	// The task that should be ran the earliest is at the end
	for (auto it = taskBatch.crbegin(); it != taskBatch.crend(); ++it) {
		_queue.push_back(*it);
	}
	
	return _queue.size();
}

Task *FIFOQueue::getTask()
{
	std::lock_guard<SpinLock> guard(_lock);
	
	if (_queue.size() == 0) {
		return nullptr;
	}
	
	Task *task = _queue.front();
	_queue.pop_front();
	
	return task;
}

std::vector<Task *> FIFOQueue::getTaskBatch(int elements)
{
	std::lock_guard<SpinLock> guard(_lock);
	
	if (elements == -1 || (size_t)elements > _queue.size()) {
		elements = _queue.size();
	}
	
	std::vector<Task *> taskBatch(elements);
	for (int i = 0; i < elements; ++i) {
		// The task that should be ran the earliest is at the end
		taskBatch[elements - i - 1] = _queue.front();
		_queue.pop_front();
	}
	
	return taskBatch;
}
