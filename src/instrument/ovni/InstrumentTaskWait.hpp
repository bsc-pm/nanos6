/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_OVNI_TASK_WAIT_HPP
#define INSTRUMENT_OVNI_TASK_WAIT_HPP


#include "instrument/api/InstrumentTaskWait.hpp"
#include "OVNITrace.hpp"

namespace Instrument {
	inline void enterTaskWait(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) char const *invocationSource,
		__attribute__((unused)) task_id_t if0TaskId,
		bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		if (taskRuntimeTransition)
			OVNI::taskWaitEnter();
	}

	inline void exitTaskWait(
		__attribute__((unused)) task_id_t taskId,
		bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		if (taskRuntimeTransition)
			OVNI::taskWaitExit();
	}
}


#endif // INSTRUMENT_OVNI_TASK_WAIT_HPP
