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
		Ovni::taskExecute(taskId._taskId);
	}

	inline void endTask(
		task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		Ovni::taskEnd(taskId._taskId);
	}

	inline void destroyTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

	inline void startTaskforCollaborator(
		__attribute__((unused)) task_id_t taskforId,
		__attribute__((unused)) task_id_t collaboratorId,
		__attribute__((unused)) bool first,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		Ovni::startTaskfor();
	}

	inline void endTaskforCollaborator(
		__attribute__((unused)) task_id_t taskforId,
		__attribute__((unused)) task_id_t collaboratorId,
		__attribute__((unused)) bool last,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		Ovni::stopTaskfor();
	}
}


#endif // INSTRUMENT_OVNI_TASK_EXECUTION_HPP
