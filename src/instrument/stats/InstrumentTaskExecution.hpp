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
		
		Instrument::Stats::PhaseInfo &phaseInfo = Stats::_threadStats->getCurrentPhaseRef();
		Instrument::Stats::TaskInfo &taskInfo = phaseInfo._perTask[taskId->_type];
		taskInfo += taskId->_times;
		taskInfo += taskId->_hardwareCounters;
		
		delete taskId;
	}
}


#endif // INSTRUMENT_STATS_TASK_EXECUTION_HPP
