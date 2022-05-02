/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_OVNI_BLOCKING_API_HPP
#define INSTRUMENT_OVNI_BLOCKING_API_HPP


#include "instrument/api/InstrumentBlockingAPI.hpp"
#include "OVNITrace.hpp"

namespace Instrument {
	inline void enterBlockCurrentTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		// TODO OVNI blocking API
	}

	inline void exitBlockCurrentTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		// TODO OVNI blocking API
	}

	inline void enterUnblockTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		// TODO OVNI blocking API
	}

	inline void exitUnblockTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		// TODO OVNI blocking API
	}

	inline void enterWaitFor(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		OVNI::waitforEnter();
	}

	inline void exitWaitFor(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		OVNI::waitforExit();
	}
}


#endif // INSTRUMENT_OVNI_BLOCKING_HPP
