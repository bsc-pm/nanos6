/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef FIFO_QUEUE_HPP
#define FIFO_QUEUE_HPP

#include <deque>

#include "lowlevel/SpinLock.hpp"
#include "../TreeSchedulerQueueInterface.hpp"

class FIFOQueue: public TreeSchedulerQueueInterface {
private:
	SpinLock _lock;
	std::deque<Task *> _queue;
public:
	~FIFOQueue();
	size_t addTask(Task *task, SchedulerInterface::ReadyTaskHint hint);
	size_t addTaskBatch(const std::vector<Task *> &taskBatch);
	Task *getTask();
	std::vector<Task *> getTaskBatch(int elements);
	size_t getSize();
};


#endif // FIFO_QUEUE
