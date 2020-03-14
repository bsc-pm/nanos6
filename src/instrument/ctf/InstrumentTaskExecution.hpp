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
	inline void startTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) InstrumentationContext const &context)
	{
	}

	inline void returnToTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) InstrumentationContext const &context)
	{
	}

	inline void endTask(task_id_t taskId, __attribute__((unused)) InstrumentationContext const &context)
	{
		CTFTaskInfo *ctfTaskInfo = taskId._ctfTaskInfo;
		assert(CTFTaskInfo != nullptr);
		CTFAPI::tracepoint(TP_NANOS6_TASK_END, static_cast<uint32_t>(ctfTaskInfo->_taskId));
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


#endif // INSTRUMENT_CTF_TASK_EXECUTION_HPP
