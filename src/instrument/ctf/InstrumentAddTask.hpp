/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_ADD_TASK_HPP
#define INSTRUMENT_CTF_ADD_TASK_HPP

#include <cassert>

#include "Nanos6CTFEvents.hpp"
#include "InstrumentTaskId.hpp"
#include "tasks/TasktypeData.hpp"

#include "../api/InstrumentAddTask.hpp"

namespace Instrument {

	inline ctf_task_type_id_t ctfGetTaskTypeId(nanos6_task_info_t *taskInfo)
	{
		assert(taskInfo->task_type_data);
		TasktypeData *tasktypeData = (TasktypeData *) taskInfo->task_type_data;
		task_type_id_t &instrumentId = tasktypeData->getInstrumentationId();
		return instrumentId.id;
	}

	inline ctf_task_type_id_t ctfAutoSetTaskTypeId(nanos6_task_info_t *taskInfo)
	{
		assert(taskInfo->task_type_data);
		TasktypeData *tasktypeData = (TasktypeData *) taskInfo->task_type_data;
		task_type_id_t &instrumentId = tasktypeData->getInstrumentationId();
		return instrumentId.autoAssingId();
	}

	inline task_id_t enterCreateTask(
		nanos6_task_info_t *taskInfo,
		__attribute__((unused)) nanos6_task_invocation_info_t *taskInvokationInfo,
		__attribute__((unused)) size_t flags,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		ctf_task_id_t taskId;
		ctf_task_type_id_t taskTypeId;

		task_id_t task_id(true);
		taskId = task_id._taskId;
		taskTypeId = ctfGetTaskTypeId(taskInfo);

		tp_task_create_enter(taskTypeId, taskId);

		return task_id;
	}

	inline void exitCreateTask()
	{
		tp_task_create_exit();
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

	inline void enterSubmitTask()
	{
		tp_task_submit_enter();
	}

	inline void exitSubmitTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		tp_task_submit_exit();
	}

	inline task_id_t enterInitTaskforCollaborator(
		__attribute__((unused)) task_id_t taskforId,
		nanos6_task_info_t *taskInfo,
		__attribute__((unused)) nanos6_task_invocation_info_t *taskInvokationInfo,
		__attribute__((unused)) size_t flags,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		ctf_task_id_t taskId;
		ctf_task_type_id_t taskTypeId;

		task_id_t task_id(true);
		taskId = task_id._taskId;
		taskTypeId = ctfGetTaskTypeId(taskInfo);
		tp_taskfor_init_enter(taskTypeId, taskId);

		return task_id;
	}

	inline void exitInitTaskforCollaborator(
		__attribute__((unused)) task_id_t taskforId,
		__attribute__((unused)) task_id_t collaboratorId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		tp_taskfor_init_exit();
	}

	inline void registeredNewSpawnedTaskType(nanos6_task_info_t *taskInfo)
	{
		const char *label = taskInfo->implementations[0].task_label;
		const char *source = taskInfo->implementations[0].declaration_source;
		ctf_task_type_id_t taskTypeId = ctfAutoSetTaskTypeId(taskInfo);
		tp_task_label(label, source, taskTypeId);
	}
}


#endif // INSTRUMENT_CTF_ADD_TASK_HPP
