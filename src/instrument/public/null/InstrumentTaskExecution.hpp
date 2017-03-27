#ifndef INSTRUMENT_NULL_TASK_EXECUTION_HPP
#define INSTRUMENT_NULL_TASK_EXECUTION_HPP


#include <InstrumentTaskId.hpp>
#include <InstrumentThreadId.hpp>

#include "../api/InstrumentCPUId.hpp"
#include "../api/InstrumentTaskExecution.hpp"


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
	
	inline void destroyTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) cpu_id_t cpuId, __attribute__((unused)) thread_id_t currentThreadId)
	{
	}
}


#endif // INSTRUMENT_NULL_TASK_EXECUTION_HPP
