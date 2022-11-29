/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_OVNI_TASK_EXECUTION_HPP
#define INSTRUMENT_OVNI_TASK_EXECUTION_HPP


#include <cassert>
#include <InstrumentInstrumentationContext.hpp>

#include "instrument/api/InstrumentTaskExecution.hpp"
#include "OvniTrace.hpp"

namespace Instrument {
	inline void startTask(
		task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		Ovni::taskBodyEnter();
		Ovni::taskExecute(taskId._taskId);
	}

	inline void endTask(
		task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		Ovni::taskEnd(taskId._taskId);
		Ovni::taskBodyExit();
	}

	inline void destroyTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
}


#endif // INSTRUMENT_OVNI_TASK_EXECUTION_HPP
