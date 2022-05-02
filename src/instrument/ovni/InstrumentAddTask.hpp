/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_OVNI_ADD_TASK_HPP
#define INSTRUMENT_OVNI_ADD_TASK_HPP

#include <cassert>

#include "OVNITrace.hpp"
#include "InstrumentTaskId.hpp"
#include "tasks/TasktypeData.hpp"

#include "instrument/api/InstrumentAddTask.hpp"

namespace Instrument {

	inline uint32_t getTaskTypeId(nanos6_task_info_t *taskInfo)
	{
		assert(taskInfo->task_type_data);
		TasktypeData *tasktypeData = (TasktypeData *) taskInfo->task_type_data;
		TasktypeInstrument &instrumentId = tasktypeData->getInstrumentationId();
		return instrumentId.id;
	}

	inline uint32_t autoSetTaskTypeId(nanos6_task_info_t *taskInfo)
	{
		assert(taskInfo->task_type_data);
		TasktypeData *tasktypeData = (TasktypeData *) taskInfo->task_type_data;
		TasktypeInstrument &instrumentId = tasktypeData->getInstrumentationId();
		return instrumentId.autoAssingId();
	}

	inline task_id_t enterCreateTask(
		nanos6_task_info_t *taskInfo,
		__attribute__((unused)) nanos6_task_invocation_info_t *taskInvokationInfo,
		__attribute__((unused)) size_t flags,
		__attribute__((unused)) bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		uint32_t taskId;
		uint32_t taskTypeId;

		task_id_t task_id(true);
		taskId = task_id._taskId;
		taskTypeId = getTaskTypeId(taskInfo);

		OVNI::createTaskEnter();
		OVNI::taskCreate(taskId, taskTypeId);

		return task_id;
	}

	inline void exitCreateTask(__attribute__((unused)) bool taskRuntimeTransition)
	{
		OVNI::createTaskExit();
	}

	inline void createdArgsBlock(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) void *argsBlockPointer,
		__attribute__((unused)) size_t originalArgsBlockSize,
		__attribute__((unused)) size_t argsBlockSize,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

	inline void createdTask(
		__attribute__((unused)) void *task,
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

	inline void enterSubmitTask(__attribute__((unused)) bool taskRuntimeTransition)
	{
		OVNI::submitEnter();
	}

	inline void exitSubmitTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		OVNI::submitExit();
	}

	inline task_id_t enterInitTaskforCollaborator(
		__attribute__((unused)) task_id_t taskforId,
		nanos6_task_info_t *taskInfo,
		__attribute__((unused)) nanos6_task_invocation_info_t *taskInvokationInfo,
		__attribute__((unused)) size_t flags,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		uint32_t taskId;
		uint32_t taskTypeId;

		task_id_t task_id(true);
		taskId = task_id._taskId;
		taskTypeId = getTaskTypeId(taskInfo);

		OVNI::createTaskEnter();
		OVNI::taskCreate(taskId, taskTypeId);

		return task_id;
	}

	inline void exitInitTaskforCollaborator(
		__attribute__((unused)) task_id_t taskforId,
		__attribute__((unused)) task_id_t collaboratorId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		OVNI::createTaskExit();
	}

	inline void registeredNewSpawnedTaskType(nanos6_task_info_t *taskInfo)
	{
		uint32_t taskTypeId = autoSetTaskTypeId(taskInfo);
		const char *label = taskInfo->implementations[0].task_type_label;
		OVNI::typeCreate(taskTypeId, label);
	}

	inline void enterSpawnFunction(__attribute__((unused)) bool taskRuntimeTransition)
	{
		OVNI::spawnFunctionEnter();
	}

	inline void exitSpawnFunction(__attribute__((unused)) bool taskRuntimeTransition)
	{
		OVNI::spawnFunctionExit();
	}
}


#endif // INSTRUMENT_OVNI_ADD_TASK_HPP
