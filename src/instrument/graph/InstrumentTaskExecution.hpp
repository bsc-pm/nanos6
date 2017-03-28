#ifndef INSTRUMENT_GRAPH_TASK_EXECUTION_HPP
#define INSTRUMENT_GRAPH_TASK_EXECUTION_HPP


#include "../api/InstrumentTaskExecution.hpp"


namespace Instrument {
	void startTask(task_id_t taskId, InstrumentationContext const &context);
	
	inline void returnToTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) InstrumentationContext const &context)
	{
	}
	
	void endTask(task_id_t taskId, InstrumentationContext const &context);
	
	inline void destroyTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) InstrumentationContext const &context)
	{
	}
}


#endif // INSTRUMENT_GRAPH_TASK_EXECUTION_HPP
