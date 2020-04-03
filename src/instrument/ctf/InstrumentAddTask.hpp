/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_ADD_TASK_HPP
#define INSTRUMENT_CTF_ADD_TASK_HPP

#include <map>

#include <CTFAPI.hpp>
#include "../api/InstrumentAddTask.hpp"
#include "InstrumentTaskId.hpp"


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
				taskLabel = "unknown";
			CTFAPI::tp_task_label(taskLabel, taskTypeId);
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

		CTFAPI::tp_task_add(taskTypeId, taskId);

		// TODO this is a workaround to identify new kind of
		// instantiated tasks (identified by its label). For each new
		// task type, we need to emit the task_label event but new found
		// task types are no reported to Nanos6, tasks are simple
		// instantiated when created. Hence, it's not possible to easily
		// distinguish when a task of a new type has been created and we
		// need to check whether the task is new every time it is added.
		// We do so by using a per-cpu map and a global map protected
		// with a spinlock.

		//uint32_t taskId;
		//uint16_t taskTypeId;
		//nanos6_task_info_t *key = taskInfo;
		//taskLabelMap_t localTaskLabelMap = &Instrument::getCPULocalData().localTaskLabelMap;

		//taskLabelMapEntry_t localTaskLabelEntry = CPU.localTaskLabelMap.emplace(key, 0);
		//taskLabelMap_t::iterator localIter = localTaskLabelEntry.first;
		//bool localFound = !localTaskLabelEntry.second;

		//if (localFound) {
		//	// if the element was in the map, retrieve the typeId value
		//	taskTypeId = localIter->second;
		//} else {
		//	// if it is the first time that this CPU sees this kind
		//	// of task (the map entry was added) ask for the
		//	// taskTypeId in the global taskTypeId mapping

		//	//check for entry in the global mmap
		//	globalTaskLabelLock.lock();

		//	taskLabelMapEntry_t globalTaskLabelEntry = globalTaskLabelMap.emplace(key, 0);
		//	taskLabelMap_t::iterator globalIter = globalTaskLabelEntry.first;
		//	bool globalFound = !globalTaskLabelEntry.second;

		//	if (globalFound) {
		//		// if exist, retrieve it.
		//		taskTypeId = globalIter->second;
		//	} else {
		//		// if not exist, request a new taskTypeId and install it
		//		taskTypeId = getNewTaskTypeId();
		//		globalIter->second = taskTypeId
		//	}

		//	globalTaskLabelLock.unlock();

		//	// install the taskTypeId into the local cpu cache
		//	localIter->second = taskTypeId;

		//	// we emit the task label tracepoint per cpu, even if
		//	// repeated, because:
		//	//   1) trace compass needs to see (globally) the task
		//	//   label events before the task instantiation events
		//	//   2) we don't want to emit the event with the lock
		//	//   held, given that it could perform a disk write.
		//	//   3) it's ok for trace compass to digest repeated
		//	//   task label events
		//	//   4) in terms of performance is not really a big deal
		//	//   5) this is still an ugly workaround to palliate the
		//	//   fact that we don't have a nanos6 entry point to
		//	//   "register" new task type id's, which should be done
		//	//   at the compiler level, so it's ok for it be uglier.
		//	//const char *taskLabel = key->implementations[0].task_label;
		//	//CTFAPI::tp_task_label(taskTypeId, taskLabel);
		//}
		//CTFAPI::tp_task_add(typeTypeId, taskId);

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
		task_id_t taskforId,
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
		CTFAPI::tp_task_add(taskTypeId, taskId);

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
