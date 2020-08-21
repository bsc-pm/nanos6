/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_ADD_TASK_HPP
#define INSTRUMENT_NULL_ADD_TASK_HPP


#include "instrument/api/InstrumentAddTask.hpp"


namespace Instrument {
	inline task_id_t enterCreateTask(
		__attribute__((unused)) nanos6_task_info_t *taskInfo,
		__attribute__((unused)) nanos6_task_invocation_info_t *taskInvokationInfo,
		__attribute__((unused)) size_t flags,
		__attribute__((unused)) bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		return task_id_t();
	}

	inline void exitCreateTask(
		__attribute__((unused)) bool taskRuntimeTransition
	) {
	}

	inline void createdArgsBlock(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) void *argsBlockPointer,
		__attribute__((unused)) size_t originalArgsBlockSize,
		__attribute__((unused)) size_t argsBlockSize,
		__attribute__((unused)) InstrumentationContext const &context)
	{
	}

	inline void createdTask(
		__attribute__((unused)) void *task,
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

	inline void enterSubmitTask(
		__attribute__((unused)) bool taskRuntimeTransition
	) {
	}

	inline void exitSubmitTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

	inline task_id_t enterInitTaskforCollaborator(
		__attribute__((unused)) task_id_t taskforId,
		__attribute__((unused)) nanos6_task_info_t *taskInfo,
		__attribute__((unused)) nanos6_task_invocation_info_t *taskInvokationInfo,
		__attribute__((unused)) size_t flags,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		return task_id_t();
	}

	inline void exitInitTaskforCollaborator(
		__attribute__((unused)) task_id_t taskforId,
		__attribute__((unused)) task_id_t collaboratorId,
		__attribute__((unused)) InstrumentationContext const &context
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
		__attribute__((unused)) bool taskRuntimeTransition
	) {
	}
}


#endif // INSTRUMENT_NULL_ADD_TASK_HPP
