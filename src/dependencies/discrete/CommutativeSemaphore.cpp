/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "CommutativeSemaphore.hpp"
#include "CPUDependencyData.hpp"
#include "DataAccessRegistration.hpp"
#include "TaskDataAccesses.hpp"
#include "tasks/Task.hpp"

#include <mutex>

CommutativeSemaphore::lock_t CommutativeSemaphore::_lock;
CommutativeSemaphore::waiting_tasks_t CommutativeSemaphore::_waitingTasks;
CommutativeSemaphore::commutative_mask_t CommutativeSemaphore::_mask;

bool CommutativeSemaphore::registerTask(Task *task)
{
	TaskDataAccesses &accessStruct = task->getDataAccesses();
	const commutative_mask_t &mask = accessStruct._commutativeMask;
	assert(mask.any());

	std::lock_guard<lock_t> guard(_lock);
	if (maskIsCompatible(mask)) {
		maskRegister(mask);
		return true;
	}

	_waitingTasks.emplace_back(std::forward<Task *>(task));
	return false;
}

void CommutativeSemaphore::releaseTask(Task *task, CPUDependencyData &hpDependencyData, ComputePlace *computePlace)
{
	TaskDataAccesses &accessStruct = task->getDataAccesses();
	const commutative_mask_t &mask = accessStruct._commutativeMask;
	assert(mask.any());
	commutative_mask_t released;

	std::lock_guard<lock_t> guard(_lock);
	maskRelease(mask);

	waiting_tasks_t::iterator it = _waitingTasks.begin();

	while (it != _waitingTasks.end()) {
		Task *candidate = *it;
		TaskDataAccesses &candidateStruct = candidate->getDataAccesses();
		const commutative_mask_t &candidateMask = candidateStruct._commutativeMask;

		if (maskIsCompatible(candidateMask)) {
			maskRegister(candidateMask);
			hpDependencyData.addSatisfiedOriginator(candidate, candidate->getDeviceType());
			assert(hpDependencyData._satisfiedOriginatorCount <= SCHEDULER_CHUNK_SIZE);

			// Ideally this should not happen here, as we are holding a lock, but it is not safe
			// to release it and grab it again without restarting the loop.
			if (hpDependencyData._satisfiedOriginatorCount == SCHEDULER_CHUNK_SIZE) {
				DataAccessRegistration::processSatisfiedOriginators(hpDependencyData, computePlace, true);
			}

			it = _waitingTasks.erase(it);

			// Keep track and cut off if we won't be releasing anything else.
			released |= (mask & candidateMask);
			if (released == mask)
				break;
		} else {
			++it;
		}
	}
}