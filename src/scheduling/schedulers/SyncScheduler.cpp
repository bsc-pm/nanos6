/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2021 Barcelona Supercomputing Center (BSC)
*/

#include "SyncScheduler.hpp"

#include <InstrumentScheduler.hpp>


Task *SyncScheduler::getTask(ComputePlace *computePlace)
{
	assert(computePlace != nullptr);

	Task *task = nullptr;
	uint64_t computePlaceIdx = computePlace->getIndex();

	Instrument::enterSchedulerLock();

	// Lock or delegate the work of getting a ready task
	if (!_lock.lockOrDelegate(computePlaceIdx, task)) {
		// Someone else acquired the lock and assigned us work
		if (task) {
			Instrument::exitSchedulerLockAsClient(task->getInstrumentationTaskId());
		} else {
			Instrument::exitSchedulerLockAsClient();
		}
		return task;
	}

	// We acquired the lock and we have to serve tasks
	Instrument::schedulerLockBecomesServer();
	setServingTasks(true);

	// The idea is to always keep a compute place inside the following scheduling loop
	// serving tasks to the rest of active compute places, except when there is work for
	// all compute places. A compute place should stay inside the scheduler to check for
	// deadline tasks but also for progressively resuming idle compute places when there
	// is available work. However, device schedulers do not work like that because they
	// already implement their progress engine using polling tasks. Also, external or
	// compute places being disabled should not serve tasks for a long time
	do {
		size_t servingIters = 0;

		// Move ready tasks from add queues to the unsynchronized scheduler
		processReadyTasks();

		// Serve the rest of computes places that are waiting
		while (servingIters < _maxServingIters && !_lock.empty()) {
			// Get the index of the waiting compute place
			computePlaceIdx = _lock.front();
			assert(computePlaceIdx < _totalComputePlaces);

			ComputePlace *waitingComputePlace = getComputePlace(computePlaceIdx);
			assert(waitingComputePlace != nullptr);

			// Try to get a ready task from the scheduler
			task = _scheduler->getReadyTask(waitingComputePlace);

			if (task != nullptr)
				Instrument::schedulerLockServesTask(task->getInstrumentationTaskId());

			// If we are using the hybrid/busy policy, avoid assigning tasks even if
			// none are found, so that threads do not spin in their body to avoid
			// contention in here. The "responsible" thread will be the one busy
			// iterating until the criteria of max busy iterations is met
			if (task != nullptr || _currentBusyIters++ >= _numBusyIters) {
				// Assign the task to the waiting compute place even if it is nullptr. The
				// responsible for serving tasks is the current compute place, and we want
				// to avoid changing the responsible constantly, as happened in the original
				// implementation
				_lock.setItem(computePlaceIdx, task);

				// Unblock the served compute place and advance to the next one
				_lock.popFront();

				_currentBusyIters = 0;
			}

			servingIters++;
			if (task == nullptr)
				break;
		}

		// No more compute places waiting; try to get work for myself
		task = _scheduler->getReadyTask(computePlace);

		// Keep serving while there is no work for the current compute
		// place or it is external/disabling
	} while (task == nullptr && !mustStopServingTasks(computePlace));

	// We are stopping to serve tasks
	setServingTasks(false);
	Instrument::exitSchedulerLockAsServer();

	// Release the lock so another compute place can serve tasks
	_lock.unlock();

	// Perform the required actions after stop serving tasks. In the case of
	// the host scheduler it should resume idle compute places to guarantee
	// that there is always a compute place serving tasks
	postServingTasks(computePlace, task);

	return task;
}
