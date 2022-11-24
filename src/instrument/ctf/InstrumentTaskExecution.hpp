/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_TASK_EXECUTION_HPP
#define INSTRUMENT_CTF_TASK_EXECUTION_HPP


#include <cassert>
#include <InstrumentInstrumentationContext.hpp>

#include "CTFTracepoints.hpp"
#include "instrument/api/InstrumentTaskExecution.hpp"


namespace Instrument {
	inline void startTask(
		task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		tp_task_start(taskId._taskId);
	}

	inline void endTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		tp_task_end();
	}

	inline void destroyTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
}


#endif // INSTRUMENT_CTF_TASK_EXECUTION_HPP
