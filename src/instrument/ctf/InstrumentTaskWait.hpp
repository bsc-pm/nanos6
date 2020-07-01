/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_TASK_WAIT_HPP
#define INSTRUMENT_CTF_TASK_WAIT_HPP


#include "../api/InstrumentTaskWait.hpp"
#include "CTFTracepoints.hpp"


namespace Instrument {
	inline void enterTaskWait(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) char const *invocationSource,
		__attribute__((unused)) task_id_t if0TaskId,
		bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		if (taskRuntimeTransition)
			tp_taskwait_tc_enter();
	}

	inline void exitTaskWait(
		__attribute__((unused)) task_id_t taskId,
		bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		if (taskRuntimeTransition)
			tp_taskwait_tc_exit();
	}
}


#endif // INSTRUMENT_CTF_TASK_WAIT_HPP
