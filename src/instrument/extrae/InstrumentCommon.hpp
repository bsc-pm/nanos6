/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_EXTRAE_COMMON_HPP
#define INSTRUMENT_EXTRAE_COMMON_HPP

#include <cassert>

#include "InstrumentExtrae.hpp"

namespace Instrument {

	inline void returnToTask(
		task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		extrae_combined_events_t ce;

		ce.HardwareCounters = 1;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 5;
		ce.nCommunications = 0;

		// Precise task count (not sampled)
		if (_detailLevel >= 1) {
			ce.nEvents += 1;
		}

		// Generate graph information
		if (_detailLevel >= 1) {
			taskId._taskInfo->_lock.lock();
			ce.nCommunications += taskId._taskInfo->_predecessors.size();
		}

		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));

		if (ce.nCommunications > 0) {
			if (ce.nCommunications < 100) {
				ce.Communications = (extrae_user_communication_t *) alloca(sizeof(extrae_user_communication_t) * ce.nCommunications);
			} else {
				ce.Communications = (extrae_user_communication_t *) malloc(sizeof(extrae_user_communication_t) * ce.nCommunications);
			}
		}

		ce.Types[0] = (extrae_type_t) EventType::RUNTIME_STATE;
		ce.Values[0] = (extrae_value_t) NANOS_RUNNING;

		nanos6_task_info_t *taskInfo = taskId._taskInfo->_taskInfo;
		assert(taskInfo != nullptr);

		ce.Types[1] = (extrae_type_t) EventType::RUNNING_CODE_LOCATION;
		ce.Values[1] = (extrae_value_t) taskInfo->implementations[0].run;

		// Use the unique taskInfo address in case it is a spawned task
		if (SpawnFunction::isSpawned(taskInfo)) {
			ce.Values[1] = (extrae_value_t) taskInfo;
		}

		ce.Types[2] = (extrae_type_t) EventType::NESTING_LEVEL;
		ce.Values[2] = (extrae_value_t) taskId._taskInfo->_nestingLevel;

		ce.Types[3] = (extrae_type_t) EventType::TASK_INSTANCE_ID;
		ce.Values[3] = (extrae_value_t) taskId._taskInfo->_taskId;

		ce.Types[4] = (extrae_type_t) EventType::PRIORITY;
		ce.Values[4] = (extrae_value_t) taskId._taskInfo->_priority;

		size_t readyTasks = --_readyTasks;

		// Precise task count (not sampled)
		if (_detailLevel >= 1) {
			ce.Types[5] = (extrae_type_t) EventType::READY_TASKS;
			ce.Values[5] = (extrae_value_t) readyTasks;

			// This counter is not so reliable, so try to skip underflows
			if (((signed long long) ce.Values[5]) < 0) {
				ce.Values[5] = 0;
			}
		}

		// Generate graph information
		if (_detailLevel >= 1) {
			int index = 0;
			for (auto const &taskAndTag : taskId._taskInfo->_predecessors) {
				ce.Communications[index].type = EXTRAE_USER_RECV;
				ce.Communications[index].tag = (extrae_comm_tag_t) taskAndTag.second;
				ce.Communications[index].size = (taskAndTag.first << 32) + taskId._taskInfo->_taskId;
				ce.Communications[index].partner = EXTRAE_COMM_PARTNER_MYSELF;
				ce.Communications[index].id = (taskAndTag.first << 32) + taskId._taskInfo->_taskId;
				index++;
			}
			taskId._taskInfo->_predecessors.clear();
			taskId._taskInfo->_lock.unlock();
		}

		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		ExtraeAPI::emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}

		if (ce.nCommunications >= 100) {
			free(ce.Communications);
		}
	}
}


#endif // INSTRUMENT_EXTRAE_COMMON_HPP
