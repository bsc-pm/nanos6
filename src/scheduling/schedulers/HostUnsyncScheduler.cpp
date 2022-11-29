/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2022 Barcelona Supercomputing Center (BSC)
*/

#include "HostUnsyncScheduler.hpp"
#include "scheduling/ready-queues/DeadlineQueue.hpp"
#include "scheduling/ready-queues/ReadyQueueDeque.hpp"
#include "scheduling/ready-queues/ReadyQueueMap.hpp"
#include "tasks/LoopGenerator.hpp"
#include "tasks/Task.hpp"

Task *HostUnsyncScheduler::getReadyTask(ComputePlace *computePlace)
{
	assert(computePlace != nullptr);
	assert(_deadlineTasks != nullptr);

	// Try to get a task with a satisfied deadline
	Task *result = _deadlineTasks->getReadyTask(computePlace);
	if (result != nullptr) {
		return result;
	}

	// Check if there is work remaining in the ready queue
	return regularGetReadyTask(computePlace);
}
