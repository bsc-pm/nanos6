/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef TREE_SCHEDULER_INTERFACE_HPP
#define TREE_SCHEDULER_INTERFACE_HPP


#include <atomic>

#include "tasks/Task.hpp"

class TreeSchedulerInterface {
public:
	virtual ~TreeSchedulerInterface()
	{
	}
	
	virtual void addTaskBatch(TreeSchedulerInterface *who, std::vector<Task *> &taskBatch, bool handleThreshold) = 0;
	virtual void updateQueueThreshold() = 0;
};


#endif // TREE_SCHEDULER_INTERFACE_HPP
