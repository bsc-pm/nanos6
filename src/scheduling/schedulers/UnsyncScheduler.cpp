/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#include "UnsyncScheduler.hpp"
#include "executors/threads/CPUManager.hpp"
#include "scheduling/ready-queues/ReadyQueueDeque.hpp"
#include "scheduling/ready-queues/ReadyQueueMap.hpp"


UnsyncScheduler::UnsyncScheduler(SchedulingPolicy policy, bool enablePriority, bool enableImmediateSuccessor)
	: _enableImmediateSuccessor(enableImmediateSuccessor), _enablePriority(enablePriority)
{
	if (enablePriority) {
		_readyTasks = new ReadyQueueMap(policy);
	} else {
		_readyTasks = new ReadyQueueDeque(policy);
	}
	
	if (enableImmediateSuccessor) {
		_immediateSuccessorTasks = std::vector<Task *>(CPUManager::getTotalCPUs(), nullptr);
	}
}

UnsyncScheduler::~UnsyncScheduler()
{
	delete _readyTasks;
}
