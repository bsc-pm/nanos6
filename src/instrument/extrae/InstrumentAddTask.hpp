/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_EXTRAE_ADD_TASK_HPP
#define INSTRUMENT_EXTRAE_ADD_TASK_HPP


#include "InstrumentExtrae.hpp"
#include "system/ompss/SpawnFunction.hpp"
#include "instrument/api/InstrumentAddTask.hpp"
#include "../support/InstrumentThreadLocalDataSupport.hpp"

#include <cassert>
#include <mutex>


class Task;


namespace Instrument {
	inline task_id_t enterCreateTask(
		nanos6_task_info_t *taskInfo,
		__attribute__((unused)) nanos6_task_invocation_info_t *taskInvokationInfo,
		__attribute__((unused)) size_t flags,
		__attribute__((unused)) bool taskRuntimeTransition,
		InstrumentationContext const &context
	) {
		size_t liveTasks = ++_liveTasks;

		ThreadLocalData &threadLocal = getThreadLocalData();
		Extrae::TaskInfo *_extraeTaskInfo = nullptr;
		if (threadLocal._nestingLevels.empty()) {
			// This may be an external thread, therefore assume that it is a spawned task
			_extraeTaskInfo = new Extrae::TaskInfo(taskInfo, 0, context._taskId._taskInfo);
		} else {
			_extraeTaskInfo = new Extrae::TaskInfo(taskInfo, threadLocal._nestingLevels.back()+1, context._taskId._taskInfo);
		}

		extrae_combined_events_t ce;

		ce.HardwareCounters = 0;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 2;
		ce.nCommunications = 0;

		// Generate graph information
		if (_detailLevel >= 1) {
			ce.nCommunications++;
		}

		// Precise task count (not sampled)
		if (_detailLevel >= 1) {
			ce.nEvents += 1;
		}

		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));

		ce.Types[0] = (extrae_type_t) EventType::RUNTIME_STATE;
		ce.Values[0] = (extrae_value_t) NANOS_CREATION;

		ce.Types[1] = (extrae_type_t) EventType::INSTANTIATING_CODE_LOCATION;
		ce.Values[1] = (extrae_value_t) taskInfo->implementations[0].run;

		// Use the unique taskInfo address in case it is a spawned task
		if (SpawnFunction::isSpawned(taskInfo)) {
			ce.Values[1] = (extrae_value_t) taskInfo;
		}

		// Precise task count (not sampled)
		if (_detailLevel >= 1) {
			ce.Types[2] = (extrae_type_t) EventType::LIVE_TASKS;
			ce.Values[2] = (extrae_value_t) liveTasks;
		}

		if (ce.nCommunications > 0) {
			ce.Communications = (extrae_user_communication_t *) alloca(sizeof(extrae_user_communication_t) * ce.nCommunications);
		}

		// Generate graph information
		if (_detailLevel >= 1) {
			ce.Communications[0].type = EXTRAE_USER_SEND;
			ce.Communications[0].tag = (extrae_comm_tag_t) instantiation_dependency_tag;
			ce.Communications[0].size = 0;
			ce.Communications[0].partner = EXTRAE_COMM_PARTNER_MYSELF;
			ce.Communications[0].id = _extraeTaskInfo->_taskId;
			_extraeTaskInfo->_predecessors.emplace(0, instantiation_dependency_tag);
		}

		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}

		assert(taskInfo != nullptr);
		{
			std::lock_guard<SpinLock> guard(_userFunctionMapLock);
			_userFunctionMap.insert(taskInfo);
		}

		ExtraeAPI::emit_CombinedEvents ( &ce );

		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}

		return task_id_t(_extraeTaskInfo);
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
		__attribute__((unused)) InstrumentationContext const &context
	) {
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
		extrae_combined_events_t ce;

		ce.HardwareCounters = 0;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 2;
		ce.nCommunications = 0;

		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));

		ce.Types[0] = (extrae_type_t) EventType::RUNTIME_STATE;
		ce.Values[0] = (extrae_value_t) NANOS_RUNNING;

		ce.Types[1] = (extrae_type_t) EventType::INSTANTIATING_CODE_LOCATION;
		ce.Values[1] = (extrae_value_t) nullptr;

		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		ExtraeAPI::emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}

	inline task_id_t enterInitTaskforCollaborator(
		__attribute__((unused)) task_id_t taskforId,
		nanos6_task_info_t *taskInfo,
		__attribute__((unused)) nanos6_task_invocation_info_t *taskInvokationInfo,
		__attribute__((unused)) size_t flags,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		ThreadLocalData &threadLocal = getThreadLocalData();
		Extrae::TaskInfo *_extraeTaskInfo = nullptr;
		if (threadLocal._nestingLevels.empty()) {
			// This may be an external thread, therefore assume that it is a spawned task
			_extraeTaskInfo = new Extrae::TaskInfo(taskInfo, 0, context._taskId._taskInfo);
		} else {
			_extraeTaskInfo = new Extrae::TaskInfo(taskInfo, threadLocal._nestingLevels.back()+1, context._taskId._taskInfo);
		}

		// When creating a regular task, we emmit two events: runtime state and code location.
		// We emmit runtime state as NANOS_CREATION and code location as the method run by the task.
		// In this case, when adding a collaborator to taskfor, we are only emmitting one event: code location.
		// We do not emmit runtime state because adding a collaborator does not actually mean creating a task,
		// since collaborators are already created at scheduler initialization. We are just setting up some
		// data structures, and so, it is not fine to emmit NANOS_CREATION.

		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}

		ExtraeAPI::emit_SimpleEvent ((extrae_type_t) EventType::INSTANTIATING_CODE_LOCATION, (extrae_value_t) taskInfo->implementations[0].run);

		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}

		return task_id_t(_extraeTaskInfo);
	}

	inline void exitInitTaskforCollaborator(
		__attribute__((unused)) task_id_t taskforId,
		__attribute__((unused)) task_id_t collaboratorId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		// As we did not changed the runtime state in "enterInitTaskforCollaborator", we do not have to restore it here.
		// Thus, emmit only code location.

		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}

		ExtraeAPI::emit_SimpleEvent ((extrae_type_t) EventType::INSTANTIATING_CODE_LOCATION, (extrae_value_t) nullptr);

		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
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


#endif // INSTRUMENT_EXTRAE_ADD_TASK_HPP
