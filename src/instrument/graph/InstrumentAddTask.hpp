/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_GRAPH_ADD_TASK_HPP
#define INSTRUMENT_GRAPH_ADD_TASK_HPP


#include "../api/InstrumentAddTask.hpp"



namespace Instrument {

	task_id_t enterCreateTask(nanos6_task_info_t *taskInfo, nanos6_task_invocation_info_t *taskInvokationInfo, size_t flags, bool taskRuntimeTransition, InstrumentationContext const &context);
	void createdArgsBlock(task_id_t taskId, void *argsBlockPointer, size_t originalArgsBlockSize, size_t argsBlockSize, InstrumentationContext const &context);
	void createdTask(void *task, task_id_t taskId, InstrumentationContext const &context);
	task_id_t enterInitTaskforCollaborator(task_id_t taskforId, nanos6_task_info_t *taskInfo, nanos6_task_invocation_info_t *taskInvokationInfo, size_t flags, InstrumentationContext const &context);
	void exitInitTaskforCollaborator(task_id_t taskforId, task_id_t collaboratorId, InstrumentationContext const &context);

	inline void exitCreateTask(
		__attribute__((unused)) bool taskRuntimeTransition
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


#endif // INSTRUMENT_GRAPH_ADD_TASK_HPP
