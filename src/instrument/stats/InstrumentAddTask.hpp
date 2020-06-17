/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_STATS_ADD_TASK_HPP
#define INSTRUMENT_STATS_ADD_TASK_HPP

#include "InstrumentStats.hpp"
#include "instrument/api/InstrumentAddTask.hpp"


namespace Instrument {

	inline task_id_t enterAddTask(
		nanos6_task_info_t *taskInfo,
		nanos6_task_invocation_info_t *,
		size_t,
		InstrumentationContext const &context
	) {
		Stats::TaskTypeAndTimes *taskTypeAndTimes = new Stats::TaskTypeAndTimes(taskInfo, (context._taskId != task_id_t()));

		return taskTypeAndTimes;
	}

	inline void createdArgsBlock(task_id_t, void *, size_t, size_t, InstrumentationContext const &)
	{
	}

	inline void createdTask(void *, task_id_t, InstrumentationContext const &)
	{
	}

	inline void exitAddTask(task_id_t, InstrumentationContext const &)
	{
	}

	inline task_id_t enterAddTaskforCollaborator(
		task_id_t,
		nanos6_task_info_t *taskInfo,
		nanos6_task_invocation_info_t *,
		size_t,
		InstrumentationContext const &context
	) {
		Stats::TaskTypeAndTimes *taskTypeAndTimes = new Stats::TaskTypeAndTimes(taskInfo, (context._taskId != task_id_t()));

		return taskTypeAndTimes;
	}

	inline void exitAddTaskforCollaborator(task_id_t, task_id_t, InstrumentationContext const &)
	{
	}

	inline void registeredNewSpawnedTaskType(
		__attribute__((unused)) nanos6_task_info_t *taskInfo
	) {
	}
}


#endif // INSTRUMENT_STATS_ADD_TASK_HPP
