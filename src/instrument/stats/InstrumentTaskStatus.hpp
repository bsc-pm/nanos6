/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_STATS_TASK_STATUS_HPP
#define INSTRUMENT_STATS_TASK_STATUS_HPP

#include <cassert>

#include "InstrumentStats.hpp"
#include "instrument/api/InstrumentTaskStatus.hpp"


namespace Instrument {
	inline void taskIsPending(task_id_t taskId, InstrumentationContext const &)
	{
		assert(taskId->_currentTimer != nullptr);

		taskId->_currentTimer->continueAt(taskId->_times._pendingTime);
		taskId->_currentTimer = &taskId->_times._pendingTime;
	}

	inline void taskIsReady(task_id_t taskId, InstrumentationContext const &)
	{
		assert(taskId->_currentTimer != nullptr);

		taskId->_currentTimer->continueAt(taskId->_times._readyTime);
		taskId->_currentTimer = &taskId->_times._readyTime;
	}

	inline void taskIsExecuting(task_id_t taskId, bool, InstrumentationContext const &)
	{
		assert(taskId->_currentTimer != nullptr);

		taskId->_currentTimer->continueAt(taskId->_times._executionTime);
		taskId->_currentTimer = &taskId->_times._executionTime;
	}

	inline void taskIsBlocked(task_id_t taskId, task_blocking_reason_t, InstrumentationContext const &)
	{
		assert(taskId->_currentTimer != nullptr);

		taskId->_currentTimer->continueAt(taskId->_times._blockedTime);
		taskId->_currentTimer = &taskId->_times._blockedTime;
	}

	inline void taskIsZombie(task_id_t taskId, InstrumentationContext const &)
	{
		assert(taskId->_currentTimer != nullptr);

		taskId->_currentTimer->continueAt(taskId->_times._zombieTime);
		taskId->_currentTimer = &taskId->_times._zombieTime;
	}

	inline void taskIsBeingDeleted(task_id_t, InstrumentationContext const &)
	{
	}

	inline void taskHasNewPriority(task_id_t, long, InstrumentationContext const &)
	{
	}

	inline void taskforCollaboratorIsExecuting(
		task_id_t,
		task_id_t collaboratorId,
		InstrumentationContext const &
	) {
		assert(collaboratorId->_currentTimer != nullptr);

		collaboratorId->_currentTimer->continueAt(collaboratorId->_times._executionTime);
		collaboratorId->_currentTimer = &collaboratorId->_times._executionTime;
	}

	inline void taskforCollaboratorStopped(
		task_id_t taskforId,
		task_id_t collaboratorId,
		InstrumentationContext const &
	) {
		assert(taskforId != nullptr);
		assert(collaboratorId->_currentTimer != nullptr);

		collaboratorId->_currentTimer->continueAt(collaboratorId->_times._zombieTime);
		collaboratorId->_currentTimer = &collaboratorId->_times._zombieTime;

		// Synchronization required
		taskforId->_lock.lock();
		taskforId->_times._executionTime += collaboratorId->_times._executionTime;
		taskforId->_lock.unlock();
	}
}


#endif // INSTRUMENT_STATS_TASK_STATUS_HPP
