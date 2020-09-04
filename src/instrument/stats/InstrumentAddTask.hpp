/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_STATS_ADD_TASK_HPP
#define INSTRUMENT_STATS_ADD_TASK_HPP

#include "InstrumentStats.hpp"
#include "instrument/api/InstrumentAddTask.hpp"


namespace Instrument {

	inline task_id_t enterCreateTask(
		nanos6_task_info_t *taskInfo,
		nanos6_task_invocation_info_t *,
		size_t,
		bool,
		InstrumentationContext const &context
	) {
		Stats::TaskTypeAndTimes *taskTypeAndTimes = new Stats::TaskTypeAndTimes(taskInfo, (context._taskId != task_id_t()));

		return taskTypeAndTimes;
	}

	inline void exitCreateTask(
		bool
	) {
	}

	inline void createdArgsBlock(
		task_id_t,
		void *,
		size_t,
		size_t,
		InstrumentationContext const &
	) {
	}

	inline void createdTask(
		void *,
		task_id_t,
		InstrumentationContext const &
	) {
	}

	inline void enterSubmitTask(
		bool
	) {
	}

	inline void exitSubmitTask(
		task_id_t,
		bool,
		InstrumentationContext const &
	) {
	}

	inline task_id_t enterInitTaskforCollaborator(
		task_id_t,
		nanos6_task_info_t *taskInfo,
		nanos6_task_invocation_info_t *,
		size_t,
		InstrumentationContext const &context
	) {
		Stats::TaskTypeAndTimes *taskTypeAndTimes = new Stats::TaskTypeAndTimes(taskInfo, (context._taskId != task_id_t()));

		return taskTypeAndTimes;
	}

	inline void exitInitTaskforCollaborator(
		task_id_t, task_id_t,
		InstrumentationContext const &
	) {
	}

	inline void registeredNewSpawnedTaskType(
		__attribute__((unused)) nanos6_task_info_t *taskInfo
	) {
	}

	inline void enterSpawnFunction(
		__attribute__((unused)) bool taskRuntimeTransition
	) {
	}

	inline void exitSpawnFunction(
		__attribute__ ((unused)) bool taskRuntimeTransition
	) {
	}
}


#endif // INSTRUMENT_STATS_ADD_TASK_HPP
