/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "CommutativeSemaphore.hpp"
#include "CPUDependencyData.hpp"
#include "tasks/Task.hpp"
#include "TaskDataAccesses.hpp"

#include <mutex>

CommutativeSemaphore::lock_t CommutativeSemaphore::_lock;
CommutativeSemaphore::queue_t CommutativeSemaphore::_queue;
CommutativeSemaphore::commutative_mask_t CommutativeSemaphore::_mask;

static inline bool mask_compatible(CommutativeSemaphore::commutative_mask_t candidate)
{
	return ((CommutativeSemaphore::_mask & candidate) == 0);
}

static inline void mask_register(CommutativeSemaphore::commutative_mask_t mask)
{
	CommutativeSemaphore::_mask |= mask;
}

static inline void mask_release(CommutativeSemaphore::commutative_mask_t mask)
{
	CommutativeSemaphore::_mask &= ~mask;
}

bool CommutativeSemaphore::registerTask(Task *task)
{
	TaskDataAccesses &accessStruct = task->getDataAccesses();
	CommutativeSemaphore::commutative_mask_t &mask = accessStruct._commutativeMask;
	assert(mask.any());

	std::lock_guard<CommutativeSemaphore::lock_t> guard(CommutativeSemaphore::_lock);
	if (mask_compatible(mask)) {
		mask_register(mask);
		return true;
	}

	CommutativeSemaphore::_queue.emplace_back(std::forward<Task *>(task));
	return false;
}

void CommutativeSemaphore::releaseTask(Task *task, CPUDependencyData &hpDependencyData)
{
	TaskDataAccesses &accessStruct = task->getDataAccesses();
	CommutativeSemaphore::commutative_mask_t &mask = accessStruct._commutativeMask;
	assert(mask.any());
	CommutativeSemaphore::commutative_mask_t released = mask;

	std::lock_guard<CommutativeSemaphore::lock_t> guard(CommutativeSemaphore::_lock);
	mask_release(mask);

	CommutativeSemaphore::queue_t::iterator it = CommutativeSemaphore::_queue.begin();

	while (it != CommutativeSemaphore::_queue.end()) {
		Task *candidate = *it;
		TaskDataAccesses &candidateStruct = candidate->getDataAccesses();
		CommutativeSemaphore::commutative_mask_t &candidateMask = candidateStruct._commutativeMask;

		if (mask_compatible(candidateMask)) {
			mask_register(candidateMask);
			hpDependencyData._satisfiedOriginators.push_back(candidate);
			it = CommutativeSemaphore::_queue.erase(it);

			// Keep track and cut off if we won't be releasing anything else.
			released |= (mask & candidateMask);
			if (released == mask)
				break;
		} else {
			++it;
		}
	}
}