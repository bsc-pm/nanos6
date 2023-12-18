/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022-2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_OVNI_ADD_TASK_HPP
#define INSTRUMENT_OVNI_ADD_TASK_HPP

#include <cassert>

#include "OvniTrace.hpp"
#include "InstrumentTaskId.hpp"
#include "InstrumentTasktypeData.hpp"
#include "instrument/api/InstrumentAddTask.hpp"
#include "tasks/TaskInfoManager.hpp"

namespace Instrument {

	inline uint32_t getTaskTypeId(nanos6_task_info_t *taskInfo)
	{
		assert(taskInfo != nullptr);
		assert(taskInfo->task_type_data);

		TaskInfoData *taskInfoData = (TaskInfoData *) taskInfo->task_type_data;
		TasktypeInstrument &instrumentId = taskInfoData->getInstrumentationId();
		return instrumentId._taskTypeId;
	}

	inline task_id_t enterCreateTask(
		nanos6_task_info_t *taskInfo,
		__attribute__((unused)) nanos6_task_invocation_info_t *taskInvokationInfo,
		__attribute__((unused)) size_t flags,
		__attribute__((unused)) bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		Ovni::enterCreateTask();

		uint32_t taskId;
		uint32_t taskTypeId;

		task_id_t task_id;
		taskId = task_id.assignNewId();
		taskTypeId = getTaskTypeId(taskInfo);

		Ovni::taskCreate(taskId, taskTypeId);

		return task_id;
	}

	inline void exitCreateTask(__attribute__((unused)) bool taskRuntimeTransition)
	{
		Ovni::exitCreateTask();
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
		Ovni::submitTaskEnter();
	}

	inline void exitSubmitTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		Ovni::submitTaskExit();
	}

	inline void registeredNewSpawnedTaskType(nanos6_task_info_t *taskInfo)
	{
		assert(taskInfo != nullptr);
		assert(taskInfo->task_type_data != nullptr);

		TaskInfoData *taskInfoData = (TaskInfoData *) taskInfo->task_type_data;
		TasktypeInstrument &instrumentId = taskInfoData->getInstrumentationId();
		uint32_t taskTypeId = instrumentId.assignNewId();

		Ovni::typeCreate(taskTypeId, taskInfoData->getTaskTypeLabel().c_str());
	}

	inline void enterSpawnFunction(__attribute__((unused)) bool taskRuntimeTransition)
	{
		Ovni::spawnFunctionEnter();
	}

	inline void exitSpawnFunction(__attribute__((unused)) bool taskRuntimeTransition)
	{
		Ovni::spawnFunctionExit();
	}
}


#endif // INSTRUMENT_OVNI_ADD_TASK_HPP
