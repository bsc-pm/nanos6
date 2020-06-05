/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_TASK_STATUS_HPP
#define INSTRUMENT_CTF_TASK_STATUS_HPP

#include "tracepoints.hpp"

#include "../api/InstrumentTaskStatus.hpp"


namespace Instrument {
	inline void taskIsPending(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

	inline void taskIsReady(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

	inline void taskIsExecuting(
		task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		tp_task_execute(taskId._taskId);
	}

	inline void taskIsBlocked(
		task_id_t taskId,
		__attribute__((unused)) task_blocking_reason_t reason,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		tp_task_block(taskId._taskId);
	}

	inline void taskIsZombie(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

	inline void taskIsBeingDeleted(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

	inline void taskHasNewPriority(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) long priority,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

	inline void taskforCollaboratorIsExecuting(
		__attribute__((unused)) task_id_t taskforId,
		task_id_t collaboratorId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		tp_task_execute(collaboratorId._taskId);
	}

	inline void taskforCollaboratorStopped(
		__attribute__((unused)) task_id_t taskforId,
		__attribute__((unused)) task_id_t collaboratorId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
}


#endif // INSTRUMENT_CTF_TASK_STATUS_HPP
