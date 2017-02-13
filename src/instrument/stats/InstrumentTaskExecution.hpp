#ifndef INSTRUMENT_STATS_TASK_EXECUTION_HPP
#define INSTRUMENT_STATS_TASK_EXECUTION_HPP


#include "../api/InstrumentTaskExecution.hpp"
#include <InstrumentTaskId.hpp>

#include "InstrumentStats.hpp"


namespace Instrument {
	inline void startTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) cpu_id_t cpuId, __attribute__((unused)) thread_id_t currentThreadId)
	{
	}
	
	inline void returnToTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) cpu_id_t cpuId, __attribute__((unused)) thread_id_t currentThreadId)
	{
	}
	
	inline void endTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) cpu_id_t cpuId, __attribute__((unused)) thread_id_t currentThreadId)
	{
	}
	
	inline void destroyTask(task_id_t taskId, __attribute__((unused)) cpu_id_t cpuId, __attribute__((unused)) thread_id_t currentThreadId)
	{
		assert(taskId->_currentTimer != 0);
		taskId->_currentTimer->stop();
		taskId->_currentTimer = 0;
		
		Stats::_threadStats->_perTask[taskId->_type] += taskId->_times;
		Stats::_threadStats->_perTask[taskId->_type] += taskId->_hardwareCounters;
		delete taskId;
	}
}


#endif // INSTRUMENT_STATS_TASK_EXECUTION_HPP
