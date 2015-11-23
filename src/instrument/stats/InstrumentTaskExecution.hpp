#ifndef INSTRUMENT_STATS_TASK_EXECUTION_HPP
#define INSTRUMENT_STATS_TASK_EXECUTION_HPP


#include "../InstrumentTaskExecution.hpp"
#include <InstrumentTaskId.hpp>

#include "InstrumentStats.hpp"


namespace Instrument {
	inline void startTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) CPU *cpu, __attribute__((unused)) WorkerThread *currentThread)
	{
	}
	
	inline void returnToTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) CPU *cpu, __attribute__((unused)) WorkerThread *currentThread)
	{
	}
	
	inline void endTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) CPU *cpu, __attribute__((unused)) WorkerThread *currentThread)
	{
	}
	
	inline void destroyTask(task_id_t taskId, __attribute__((unused)) CPU *cpu, __attribute__((unused)) WorkerThread *currentThread)
	{
		assert(taskId->_currentTimer != 0);
		taskId->_currentTimer->stop();
		taskId->_currentTimer = 0;
		
		Stats::_threadStats->_perTask[taskId->_type] += taskId->_times;
		delete taskId;
	}
}


#endif // INSTRUMENT_STATS_TASK_EXECUTION_HPP
