/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_TASK_WAIT_HPP
#define INSTRUMENT_TASK_WAIT_HPP


#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>


class Task;


namespace Instrument {
	//! This function is called upon entering the task wait API
	//! If taskRuntimeTransistion is true, Task Hardware Counters have been updated before calling this function.
	//! \param[in] taskID The waiting task identifier
	//! \param[in] invocationSource Source location starting that called this API
	//! \param[in] if0TaskId If0 task Identifier (if called from If0Task only)
	//! \param[in] taskRuntimeTransition whether this API function was called from task or
	//! runtime code
	void enterTaskWait(task_id_t taskId, char const *invocationSource, task_id_t if0TaskId, bool taskRuntimeTransition, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());

	//! This function is called upon entering the task wait API
	//! If taskRuntimeTransistion is true, Runtime Hardware Counters have been updated before calling this function.
	//! \param[in] taskID The waiting task identifier
	//! \param[in] taskRuntimeTransition whether this API function was called from task or
	//! runtime code
	void exitTaskWait(task_id_t taskId, bool taskRuntimeTransition, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
}


#endif // INSTRUMENT_TASK_WAIT_HPP
