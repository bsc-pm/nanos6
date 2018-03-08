/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#include "LIFOQueue.hpp"

LIFOQueue::~LIFOQueue()
{
	assert(_queue.size() == 0);
}

size_t LIFOQueue::addTask(Task *task, __attribute__((unused)) SchedulerInterface::ReadyTaskHint hint)
{
	std::lock_guard<SpinLock> guard(_lock);
	
	_queue.push_front(task);
	
	return _queue.size();
}

size_t LIFOQueue::addTaskBatch(const std::vector<Task *> &taskBatch)
{
	std::lock_guard<SpinLock> guard(_lock);
	
	// The task that should be ran the earliest is at the end
	for (auto it = taskBatch.crbegin(); it != taskBatch.crend(); ++it) {
		_queue.push_front(*it);
	}
	
	return _queue.size();
}

Task *LIFOQueue::getTask()
{
	std::lock_guard<SpinLock> guard(_lock);
	
	if (_queue.size() == 0) {
		return nullptr;
	}
	
	Task *task = _queue.front();
	_queue.pop_front();
	
	return task;
}

std::vector<Task *> LIFOQueue::getTaskBatch(int elements)
{
	std::lock_guard<SpinLock> guard(_lock);
	
	if (elements == -1 || (size_t)elements > _queue.size()) {
		elements = _queue.size();
	}
	
	std::vector<Task *> taskBatch(elements);
	for (int i = 0; i < elements; ++i) {
		// The task that should be ran the earliest is at the end of the vector
		// Get the oldest added tasks (the ones that will run the latest)
		taskBatch[i] = _queue.back();
		_queue.pop_back();
	}
	
	return taskBatch;
}
