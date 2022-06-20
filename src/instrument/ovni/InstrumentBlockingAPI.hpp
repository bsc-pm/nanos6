/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_OVNI_BLOCKING_API_HPP
#define INSTRUMENT_OVNI_BLOCKING_API_HPP


#include "instrument/api/InstrumentBlockingAPI.hpp"
#include "OvniTrace.hpp"

namespace Instrument {
	inline void enterBlockCurrentTask(
		task_id_t taskId,
		__attribute__((unused)) bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		Ovni::taskPause(taskId._taskId);
		Ovni::pauseEnter();
	}

	inline void exitBlockCurrentTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		Ovni::pauseExit();
		Ovni::taskResume(taskId._taskId);
	}

	inline void enterUnblockTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

	inline void exitUnblockTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

	inline void enterWaitFor(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		Ovni::waitforEnter();
	}

	inline void exitWaitFor(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		Ovni::waitforExit();
	}
}


#endif // INSTRUMENT_OVNI_BLOCKING_HPP
