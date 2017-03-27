#ifndef INSTRUMENT_STATS_ADD_TASK_HPP
#define INSTRUMENT_STATS_ADD_TASK_HPP


#include "../api/InstrumentAddTask.hpp"

#include "InstrumentStats.hpp"


namespace Instrument {
	
	inline task_id_t enterAddTask(
		nanos_task_info *taskInfo,
		__attribute__((unused)) nanos_task_invocation_info *taskInvokationInfo,
		__attribute__((unused)) size_t flags,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		Stats::TaskTypeAndTimes *taskTypeAndTimes = new Stats::TaskTypeAndTimes(taskInfo, (context._taskId != task_id_t()));
		return taskTypeAndTimes;
	}
	
	inline void createdTask(__attribute__((unused)) void *task,
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline void exitAddTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
}


#endif // INSTRUMENT_STATS_ADD_TASK_HPP
