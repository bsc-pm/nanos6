#ifndef INSTRUMENT_GRAPH_TASK_EXECUTION_HPP
#define INSTRUMENT_GRAPH_TASK_EXECUTION_HPP


#include <../InstrumentCPUId.hpp>
#include <InstrumentTaskId.hpp>
#include <InstrumentThreadId.hpp>

#include "../InstrumentTaskExecution.hpp"


namespace Instrument {
	void startTask(task_id_t taskId, cpu_id_t cpuId, thread_id_t currentThreadId);
	
	inline void returnToTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) cpu_id_t cpuId, __attribute__((unused)) thread_id_t currentThreadId)
	{
	}
	
	void endTask(task_id_t taskId, cpu_id_t cpuId, thread_id_t currentThreadId);
	
	inline void destroyTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) cpu_id_t cpuId, __attribute__((unused)) thread_id_t currentThreadId)
	{
	}
}


#endif // INSTRUMENT_GRAPH_TASK_EXECUTION_HPP
