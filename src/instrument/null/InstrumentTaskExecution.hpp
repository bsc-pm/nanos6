#ifndef INSTRUMENT_NULL_TASK_EXECUTION_HPP
#define INSTRUMENT_NULL_TASK_EXECUTION_HPP


#include "../InstrumentTaskExecution.hpp"
#include <InstrumentTaskId.hpp>


namespace Instrument {
	inline void startTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) CPU *cpu, __attribute__((unused)) WorkerThread *currentThread)
	{
	}
	
	inline void returnToTask(__attribute__((unused)) task_id_t taskIdk, __attribute__((unused)) CPU *cpu, __attribute__((unused)) WorkerThread *currentThread)
	{
	}
	
	inline void endTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) CPU *cpu, __attribute__((unused)) WorkerThread *currentThread)
	{
	}
	
}


#endif // INSTRUMENT_NULL_TASK_EXECUTION_HPP
