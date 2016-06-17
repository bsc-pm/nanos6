#ifndef INSTRUMENT_TASK_EXECUTION_HPP
#define INSTRUMENT_TASK_EXECUTION_HPP


#include <InstrumentTaskId.hpp>
#include <InstrumentThreadId.hpp>

#include "InstrumentCPUId.hpp"


namespace Instrument {
	void startTask(task_id_t taskId, cpu_id_t cpuId, thread_id_t currentThreadId);
	void returnToTask(task_id_t taskId, cpu_id_t cpuId, thread_id_t currentThreadId);
	void endTask(task_id_t taskId, cpu_id_t cpuId, thread_id_t currentThreadId);
	void destroyTask(task_id_t taskId, cpu_id_t cpuId, thread_id_t currentThreadId);
}


#endif // INSTRUMENT_TASK_EXECUTION_HPP
