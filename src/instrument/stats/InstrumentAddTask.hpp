#ifndef INSTRUMENT_STATS_ADD_TASK_HPP
#define INSTRUMENT_STATS_ADD_TASK_HPP


#include "../InstrumentAddTask.hpp"
#include <InstrumentTaskId.hpp>

#include "InstrumentStats.hpp"


class Task;


namespace Instrument {
	
	inline task_id_t enterAddTask(nanos_task_info *taskInfo, __attribute__((unused)) nanos_task_invocation_info *taskInvokationInfo)
	{
		Stats::TaskTypeAndTimes *taskTypeAndTimes = new Stats::TaskTypeAndTimes(taskInfo);
		return taskTypeAndTimes;
	}
	
	inline void createdTask(__attribute__((unused)) Task *task, __attribute__((unused)) task_id_t taskId)
	{
	}
	
	inline void exitAddTask(__attribute__((unused)) task_id_t taskId)
	{
	}
	
}


#endif // INSTRUMENT_STATS_ADD_TASK_HPP
