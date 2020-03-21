/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include "SyncScheduler.hpp"
#include "scheduling/schedulers/device/DeviceScheduler.hpp"
#include "scheduling/schedulers/device/SubDeviceScheduler.hpp"

Task *SyncScheduler::getTask(ComputePlace *computePlace)
{
	Task *task = nullptr;

	// Special case for device polling services that get ready tasks
	if (computePlace == nullptr) {
		_lock.lock();
		// Move all tasks from addQueues to the ready queue
		processReadyTasks();
		task = _scheduler->getReadyTask(computePlace);
		_lock.unsubscribe();
		assert(task == nullptr || task->isRunnable());
		return task;
	}

	assert(computePlace != nullptr);

	uint64_t const currentComputePlaceIndex = computePlace->getIndex();

	// Subscribe to the lock
	uint64_t const ticket = _lock.subscribeOrLock(currentComputePlaceIndex);

	if (getAssignedTask(currentComputePlaceIndex, ticket, task)) {
		// Someone got the lock and gave me work to do
		return task;
	}

	// We acquired the lock and we move all tasks from addQueues to the ready queue
	processReadyTasks();

	uint64_t waitingComputePlaceIndex;
	uint64_t i = ticket + 1;

	// Serve all the subscribers, while there is work to give them
	while (_lock.popWaitingCPU(i, waitingComputePlaceIndex)) {
#ifndef NDEBUG
		size_t numCPUs = CPUManager::getTotalCPUs();
		assert(waitingComputePlaceIndex < numCPUs);
#endif
		ComputePlace *resultComputePlace = getComputePlace(_deviceType, waitingComputePlaceIndex);
		assert(resultComputePlace != nullptr);

		task = _scheduler->getReadyTask(resultComputePlace);
		if (task == nullptr)
			break;

		// Put a task into the subscriber slot
		assignTask(waitingComputePlaceIndex, i, task);

		// Advance the ticket of the subscriber just served
		_lock.unsubscribe();
		i++;
	}

	// No more subscribers; try to get work for myself
	task = _scheduler->getReadyTask(computePlace);
	_lock.unsubscribe();

	return task;
}
