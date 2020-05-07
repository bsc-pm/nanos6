/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_ADD_TASK_HPP
#define INSTRUMENT_CTF_ADD_TASK_HPP

#include <map>

#include "Nanos6CTFEvents.hpp"
#include "InstrumentTaskId.hpp"

#include "../api/InstrumentAddTask.hpp"

namespace Instrument {

	inline ctf_task_type_id_t getTaskTypeID(nanos6_task_info_t *key) {
		ctf_task_type_id_t taskTypeId;

		globalTaskLabelLock.lock();

		taskLabelMapEntry_t globalTaskLabelEntry = globalTaskLabelMap.emplace(key, 0);
		taskLabelMap_t::iterator globalIter = globalTaskLabelEntry.first;
		bool globalFound = !globalTaskLabelEntry.second;

		if (globalFound) {
			// if exist, retrieve it.
			taskTypeId = globalIter->second;
		} else {
			// if not exist, request a new taskTypeId and install it
			taskTypeId = getNewTaskTypeId();
			globalIter->second = taskTypeId;
			char *taskLabel = (char *) key->implementations[0].task_label;
			if (taskLabel == nullptr)
				taskLabel = (char *) key->implementations[0].declaration_source;
			tp_task_label(taskLabel, taskTypeId);
		}

		globalTaskLabelLock.unlock();

		return taskTypeId;
	}

	inline task_id_t enterAddTask(
		nanos6_task_info_t *taskInfo,
		__attribute__((unused)) nanos6_task_invocation_info_t *taskInvokationInfo,
		__attribute__((unused)) size_t flags,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		ctf_task_id_t taskId;
		ctf_task_type_id_t taskTypeId;

		nanos6_task_info_t *key = taskInfo;
		task_id_t task_id(true);
		taskId = task_id._taskId;
		taskTypeId = getTaskTypeID(key);

		tp_task_add(taskTypeId, taskId);

		return task_id;
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

	inline void exitAddTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

	inline task_id_t enterAddTaskforCollaborator(
		__attribute__((unused)) task_id_t taskforId,
		nanos6_task_info_t *taskInfo,
		__attribute__((unused)) nanos6_task_invocation_info_t *taskInvokationInfo,
		__attribute__((unused)) size_t flags,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		ctf_task_id_t taskId;
		ctf_task_type_id_t taskTypeId;

		nanos6_task_info_t *key = taskInfo;
		task_id_t task_id(true);
		taskId = task_id._taskId;
		taskTypeId = getTaskTypeID(key);
		tp_task_add(taskTypeId, taskId);

		return task_id;
	}

	inline void exitAddTaskforCollaborator(
		__attribute__((unused)) task_id_t taskforId,
		__attribute__((unused)) task_id_t collaboratorId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
}


#endif // INSTRUMENT_CTF_ADD_TASK_HPP
