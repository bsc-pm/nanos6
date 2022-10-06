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
		Ovni::blockEnter();
		Ovni::taskPause(taskId._taskId);
	}

	inline void exitBlockCurrentTask(
		task_id_t taskId,
		__attribute__((unused)) bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		Ovni::taskResume(taskId._taskId);
		Ovni::blockExit();
	}

	inline void enterUnblockTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		Ovni::unblockEnter();
	}

	inline void exitUnblockTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		Ovni::unblockExit();
	}

	inline void enterWaitFor(
		task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		Ovni::waitForEnter();
		Ovni::taskPause(taskId._taskId);
	}

	inline void exitWaitFor(
		task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		Ovni::taskResume(taskId._taskId);
		Ovni::waitForExit();
	}
}


#endif // INSTRUMENT_OVNI_BLOCKING_HPP
