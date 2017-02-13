#ifndef INSTRUMENT_STATS_TASK_STATUS_HPP
#define INSTRUMENT_STATS_TASK_STATUS_HPP


#include "../api/InstrumentTaskStatus.hpp"
#include <InstrumentTaskId.hpp>

#include "InstrumentStats.hpp"

#include <cassert>


namespace Instrument {
	inline void taskIsPending(task_id_t taskId)
	{
		assert(taskId->_currentTimer != 0);
		
		taskId->_currentTimer->continueAt(taskId->_times._pendingTime);
		taskId->_currentTimer = &taskId->_times._pendingTime;
	}
	
	inline void taskIsReady(task_id_t taskId)
	{
		assert(taskId->_currentTimer != 0);
		
		taskId->_currentTimer->continueAt(taskId->_times._readyTime);
		taskId->_currentTimer = &taskId->_times._readyTime;
	}
	
	inline void taskIsExecuting(task_id_t taskId)
	{
		assert(taskId->_currentTimer != 0);
		
		taskId->_currentTimer->continueAt(taskId->_times._executionTime);
		taskId->_currentTimer = &taskId->_times._executionTime;
	}
	
	inline void taskIsBlocked(task_id_t taskId, __attribute__((unused)) task_blocking_reason_t reason)
	{
		
		assert(taskId->_currentTimer != 0);
		taskId->_currentTimer->continueAt(taskId->_times._blockedTime);
		taskId->_currentTimer = &taskId->_times._blockedTime;
	}
	
	inline void taskIsZombie(task_id_t taskId)
	{
		
		assert(taskId->_currentTimer != 0);
		taskId->_currentTimer->continueAt(taskId->_times._zombieTime);
		taskId->_currentTimer = &taskId->_times._zombieTime;
	}
	
}


#endif // INSTRUMENT_STATS_TASK_STATUS_HPP
