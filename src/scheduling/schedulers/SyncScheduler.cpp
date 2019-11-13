/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "SyncScheduler.hpp"
#include "scheduling/schedulers/device/DeviceScheduler.hpp"
#include "scheduling/schedulers/device/SubDeviceScheduler.hpp"

Task *SyncScheduler::getTask(ComputePlace *computePlace, ComputePlace *deviceComputePlace)
{
	Task *task = nullptr;
	ComputePlace *deviceComputePlaceOrComputePlace =
		(deviceComputePlace != nullptr) ? deviceComputePlace : computePlace;

	// Special case for device polling services that get ready tasks.
	if (computePlace == nullptr) {
		_lock.lock();
		task = _scheduler->getReadyTask(deviceComputePlaceOrComputePlace);
		_lock.unsubscribe();
		assert(task == nullptr || task->isRunnable());
		return task;
	}

	assert(computePlace != nullptr);
	assert(computePlace->getType() == nanos6_host_device);

	uint64_t const currentCPUIndex = computePlace->getIndex();
	setRelatedComputePlace(currentCPUIndex, deviceComputePlace);

	// Subscribe to the lock.
	uint64_t const ticket = _lock.subscribeOrLock(currentCPUIndex);

	if (getAssignedTask(currentCPUIndex, ticket, task)) {
		// Someone got the lock and gave me work to do.
		assert(task->isRunnable());
		return task;
	}

	// I own the lock!
	// First of all, get all the tasks in the addQueues into the ready queue.
	processReadyTasks();

	uint64_t waitingCPUIndex;
	uint64_t i = ticket + 1;

	// Serve all the subscribers, while there is work to give them.
	while (_lock.popWaitingCPU(i, waitingCPUIndex)) {
#ifndef NDEBUG
		size_t numCPUs = CPUManager::getTotalCPUs();
		assert(waitingCPUIndex < numCPUs);
#endif
		ComputePlace *resultComputePlace = getRelatedComputePlace(waitingCPUIndex);
		assert(resultComputePlace != nullptr);

		task = _scheduler->getReadyTask(resultComputePlace);
		if (task == nullptr)
			break;

		assert(task->isRunnable());

		setRelatedComputePlace(waitingCPUIndex, nullptr);

		// Put a task into the subscriber slot.
		assignTask(waitingCPUIndex, i, task);

		// Advance the ticket of the subscriber just served.
		_lock.unsubscribe();
		i++;
	}

	// No more subscribers. Try to get work for myself.
	task = _scheduler->getReadyTask(deviceComputePlaceOrComputePlace);
	_lock.unsubscribe();

	setRelatedComputePlace(currentCPUIndex, nullptr);

	assert(task == nullptr || task->isRunnable());
	return task;
}
