/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_LINT_TASK_EXECUTION_HPP
#define INSTRUMENT_LINT_TASK_EXECUTION_HPP


#include <InstrumentInstrumentationContext.hpp>
#include <Callbacks.hpp>

#include "instrument/api/InstrumentTaskExecution.hpp"



namespace Instrument {
	inline void startTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context)
	{
		nanos6_lint_on_task_start(taskId);
	}

	inline void returnToTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context)
	{
	}

	inline void endTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context)
	{
		nanos6_lint_on_task_end(taskId);
	}

	inline void destroyTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context)
	{
		nanos6_lint_on_task_destruction(taskId);
	}

	inline void startTaskforCollaborator(
		__attribute__((unused)) task_id_t taskforId,
		__attribute__((unused)) task_id_t collaboratorId,
		__attribute__((unused)) bool first,
		__attribute__((unused)) InstrumentationContext const &context)
	{
	}

	inline void endTaskforCollaborator(
		__attribute__((unused)) task_id_t taskforId,
		__attribute__((unused)) task_id_t collaboratorId,
		__attribute((unused)) bool last,
		__attribute__((unused)) InstrumentationContext const &context)
	{
	}
}


#endif // INSTRUMENT_LINT_TASK_EXECUTION_HPP
