/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef TREE_SCHEDULER_QUEUE_INTERFACE_HPP
#define TREE_SCHEDULER_QUEUE_INTERFACE_HPP

#include <vector>

#include "../../SchedulerInterface.hpp"
#include "tasks/Task.hpp"

class TreeSchedulerQueueInterface {
public:
	static TreeSchedulerQueueInterface *initialize();

	virtual ~TreeSchedulerQueueInterface()
	{
	}
	
	virtual size_t addTask(Task *task, SchedulerInterface::ReadyTaskHint hint) = 0;
	virtual size_t addTaskBatch(const std::vector<Task *> &taskBatch) = 0;
	virtual Task *getTask() = 0;
	virtual std::vector<Task *> getTaskBatch(int elements) = 0;
	virtual size_t getSize() = 0;
};


#endif // TREE_SCHEDULER_QUEUE_INTERFACE_HPP
