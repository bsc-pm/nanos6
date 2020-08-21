/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_LINT_TASK_WAIT_HPP
#define INSTRUMENT_LINT_TASK_WAIT_HPP

#include <Callbacks.hpp>

#include "instrument/api/InstrumentTaskWait.hpp"



namespace Instrument {
	inline void enterTaskWait(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) char const *invocationSource,
		__attribute__((unused)) task_id_t if0TaskId,
		__attribute__((unused)) bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		nanos6_lint_on_taskwait_enter(taskId, invocationSource, if0TaskId);
	}

	inline void exitTaskWait(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		nanos6_lint_on_taskwait_exit(taskId);
	}
}


#endif // INSTRUMENT_LINT_TASK_WAIT_HPP
