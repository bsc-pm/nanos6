/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_ADD_TASK_HPP
#define INSTRUMENT_CTF_ADD_TASK_HPP

#include <cassert>

#include "CTFTracepoints.hpp"
#include "InstrumentTaskId.hpp"
#include "InstrumentTasktypeData.hpp"
#include "instrument/api/InstrumentAddTask.hpp"
#include "tasks/TaskInfoManager.hpp"

namespace Instrument {

	inline ctf_tasktype_id_t ctfGetTaskTypeId(nanos6_task_info_t *taskInfo)
	{
		assert(taskInfo->task_type_data);
		TaskInfoData *taskInfoData = (TaskInfoData *) taskInfo->task_type_data;
		TasktypeInstrument &instrumentId = taskInfoData->getInstrumentationId();
		return instrumentId.id;
	}

	inline task_id_t enterCreateTask(
		nanos6_task_info_t *taskInfo,
		__attribute__((unused)) nanos6_task_invocation_info_t *taskInvokationInfo,
		__attribute__((unused)) size_t flags,
		bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		ctf_task_id_t taskId;
		ctf_tasktype_id_t taskTypeId;

		task_id_t task_id(true);
		taskId = task_id._taskId;
		taskTypeId = ctfGetTaskTypeId(taskInfo);

		if (taskRuntimeTransition) {
			tp_task_create_tc_enter(taskTypeId, taskId);
		} else {
			tp_task_create_oc_enter(taskTypeId, taskId);
		}

		return task_id;
	}

	inline void exitCreateTask(bool taskRuntimeTransition)
	{
		if (taskRuntimeTransition) {
			tp_task_create_tc_exit();
		} else {
			tp_task_create_oc_exit();
		}
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

	inline void enterSubmitTask(bool taskRuntimeTransition)
	{
		if (taskRuntimeTransition) {
			tp_task_submit_tc_enter();
		} else {
			tp_task_submit_oc_enter();
		}
	}

	inline void exitSubmitTask(
		__attribute__((unused)) task_id_t taskId,
		bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		if (taskRuntimeTransition) {
			tp_task_submit_tc_exit();
		} else {
			tp_task_submit_oc_exit();
		}
	}

	inline void registeredNewSpawnedTaskType(nanos6_task_info_t *taskInfo)
	{
		assert(taskInfo != nullptr);
		assert(taskInfo->task_type_data != nullptr);

		TaskInfoData *taskInfoData = (TaskInfoData *) taskInfo->task_type_data;
		TasktypeInstrument &instrumentId = taskInfoData->getInstrumentationId();
		ctf_tasktype_id_t taskTypeId = instrumentId.autoAssignId();

		tp_task_label(taskInfoData->getTaskTypeLabel().c_str(),
			taskInfoData->getTaskDeclarationSource().c_str(), taskTypeId);
	}

	inline void enterSpawnFunction(bool taskRuntimeTransition)
	{
		if (taskRuntimeTransition) {
			tp_spawn_function_tc_enter();
		} else {
			tp_spawn_function_oc_enter();
		}
	}

	inline void exitSpawnFunction(bool taskRuntimeTransition)
	{
		if (taskRuntimeTransition) {
			tp_spawn_function_tc_exit();
		} else {
			tp_spawn_function_oc_exit();
		}
	}
}


#endif // INSTRUMENT_CTF_ADD_TASK_HPP
