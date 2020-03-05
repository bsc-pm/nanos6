/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_TASK_EXECUTION_HPP
#define INSTRUMENT_CTF_TASK_EXECUTION_HPP


#include <cassert>
#include <CTFAPI.hpp>
#include <InstrumentInstrumentationContext.hpp>

#include "../api/InstrumentTaskExecution.hpp"


namespace Instrument {
	inline void startTask(task_id_t taskId, __attribute__((unused)) InstrumentationContext const &context)
	{
		CTFTaskInfo *ctfTaskInfo = taskId._ctfTaskInfo;
		assert(CTFTaskInfo != nullptr);
		nanos6_task_info_t *nanos6TaskInfo = ctfTaskInfo->_nanos6TaskInfo;
		assert(nanos6TaskInfo != nullptr);
		CTFAPI::tp_task_start((uint64_t) nanos6TaskInfo->implementations[0].run, ctfTaskInfo->_taskId);
	}

	inline void returnToTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) InstrumentationContext const &context)
	{
	}

	inline void endTask(task_id_t taskId, __attribute__((unused)) InstrumentationContext const &context)
	{
		CTFTaskInfo *ctfTaskInfo = taskId._ctfTaskInfo;
		assert(CTFTaskInfo != nullptr);
		CTFAPI::tp_task_stop(ctfTaskInfo->_taskId);
	}

	inline void destroyTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) InstrumentationContext const &context)
	{
	}

	inline void startTaskforCollaborator(__attribute__((unused)) task_id_t taskforId, __attribute__((unused)) task_id_t collaboratorId, __attribute__((unused)) bool first, __attribute__((unused)) InstrumentationContext const &context)
	{
	}

	inline void endTaskforCollaborator(__attribute__((unused)) task_id_t taskforId, __attribute__((unused)) task_id_t collaboratorId, __attribute((unused)) bool last, __attribute__((unused)) InstrumentationContext const &context)
	{
	}
}


#endif // INSTRUMENT_NULL_TASK_EXECUTION_HPP
