/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_STATS_TASK_STATUS_HPP
#define INSTRUMENT_STATS_TASK_STATUS_HPP


#include "../api/InstrumentTaskStatus.hpp"

#include "InstrumentStats.hpp"

#include <cassert>


namespace Instrument {
	inline void taskIsPending(
		task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context)
	{
		assert(taskId->_currentTimer != 0);
		
		taskId->_currentTimer->continueAt(taskId->_times._pendingTime);
		taskId->_currentTimer = &taskId->_times._pendingTime;
	}
	
	inline void taskIsReady(
		task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context)
	{
		assert(taskId->_currentTimer != 0);
		
		taskId->_currentTimer->continueAt(taskId->_times._readyTime);
		taskId->_currentTimer = &taskId->_times._readyTime;
	}
	
	inline void taskIsExecuting(
		task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context)
	{
		assert(taskId->_currentTimer != 0);
		
		taskId->_currentTimer->continueAt(taskId->_times._executionTime);
		taskId->_currentTimer = &taskId->_times._executionTime;
		
		taskId->_hardwareCounters.start();
	}
	
	inline void taskIsBlocked(
		task_id_t taskId, __attribute__((unused)) task_blocking_reason_t reason,
		__attribute__((unused)) InstrumentationContext const &context)
	{
		taskId->_hardwareCounters.accumulateAndStop();
		
		assert(taskId->_currentTimer != 0);
		taskId->_currentTimer->continueAt(taskId->_times._blockedTime);
		taskId->_currentTimer = &taskId->_times._blockedTime;
	}
	
	inline void taskIsZombie(
		task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context)
	{
		taskId->_hardwareCounters.accumulateAndStop();
		
		assert(taskId->_currentTimer != 0);
		taskId->_currentTimer->continueAt(taskId->_times._zombieTime);
		taskId->_currentTimer = &taskId->_times._zombieTime;
	}
	
	inline void taskIsBeingDeleted(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline void taskHasNewPriority(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) long priority,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
}


#endif // INSTRUMENT_STATS_TASK_STATUS_HPP
