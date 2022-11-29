/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2022 Barcelona Supercomputing Center (BSC)
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
}


#endif // INSTRUMENT_LINT_TASK_EXECUTION_HPP
