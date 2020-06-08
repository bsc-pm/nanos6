/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include "SyncScheduler.hpp"

Task *SyncScheduler::getTask(ComputePlace *computePlace)
{
	assert(computePlace != nullptr);

	Task *task = nullptr;
	uint64_t computePlaceIndex = computePlace->getIndex();

	// Subscribe to the lock
	uint64_t ticket = _lock.subscribeOrLock(computePlaceIndex);
	if (getAssignedTask(computePlaceIndex, ticket, task)) {
		// Someone got the lock and assigned work to do
		return task;
	}

	// We acquired the lock so we have to serve tasks
	setServingTasks(true);
	ticket++;

	// The idea is to always keep a compute place inside the following scheduling loop
	// serving tasks to the rest of active compute places, except when there is work for
	// all compute places. A compute place should stay inside the scheduler to check for
	// deadline tasks but also for progressively resuming idle compute places when there
	// is available work. However, device scheduler do not work like that because they
	// already implement their progress engine using polling services. Also, external
	// or compute places being disabled should not serve tasks for a long time
	do {
		// Move ready tasks from add queues to the unsynchronized scheduler
		processReadyTasks();

		// Serve the subscribers that are waiting
		while (_lock.popWaitingCPU(ticket, computePlaceIndex)) {
			assert(computePlaceIndex < _totalComputePlaces);

			ComputePlace *waitingComputePlace = getComputePlace(computePlaceIndex);
			assert(waitingComputePlace != nullptr);

			// Try to get a ready task from the scheduler
			task = _scheduler->getReadyTask(waitingComputePlace);

			// Assign the task to the subscriber slot even if it nullptr. The
			// responsible for serving tasks is the current compute place, and
			// we want to avoid changing the responsible constantly (as happened
			// in the original implementation)
			assignTask(computePlaceIndex, ticket, task);

			// Advance the ticket of the subscriber just served
			_lock.unsubscribe();
			ticket++;

			if (task == nullptr)
				break;
		}

		// No more subscribers; try to get work for myself
		task = _scheduler->getReadyTask(computePlace);

		// Keep serving while there is no work for the current compute
		// place or it is external/disabling
	} while (task == nullptr && !mustStopServingTasks(computePlace));

	setServingTasks(false);
	_lock.unsubscribe();

	// Perform the required actions after stop serving tasks. In the case
	// of the host scheduler it should resume idle compute places to guarantee
	// that there is always a compute place serving tasks
	postServingTasks(computePlace, task);

	return task;
}
