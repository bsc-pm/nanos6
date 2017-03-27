#ifndef INSTRUMENT_NULL_ADD_TASK_HPP
#define INSTRUMENT_NULL_ADD_TASK_HPP


#include "../api/InstrumentAddTask.hpp"
#include <InstrumentTaskId.hpp>


namespace Instrument {
	inline task_id_t enterAddTask(__attribute__((unused)) nanos_task_info *taskInfo, __attribute__((unused)) nanos_task_invocation_info *taskInvokationInfo)
	{
		return task_id_t();
	}
	
	inline void createdTask(__attribute__((unused)) void *task, __attribute__((unused)) task_id_t taskId)
	{
	}
	
	inline void exitAddTask(__attribute__((unused)) task_id_t taskId)
	{
	}
	
}


#endif // INSTRUMENT_NULL_ADD_TASK_HPP
