/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef LIFO_QUEUE_HPP
#define LIFO_QUEUE_HPP

#include <deque>

#include "lowlevel/SpinLock.hpp"
#include "../TreeSchedulerQueueInterface.hpp"

class LIFOQueue: public TreeSchedulerQueueInterface {
private:
	SpinLock _lock;
	std::deque<Task *> _queue;
public:
	~LIFOQueue();
	size_t addTask(Task *task, SchedulerInterface::ReadyTaskHint hint);
	size_t addTaskBatch(const std::vector<Task *> &taskBatch);
	Task *getTask();
	std::vector<Task *> getTaskBatch(int elements);
	size_t getSize();
};


#endif // LIFO_QUEUE
