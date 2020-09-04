/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2018-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_BLOCKING_API_HPP
#define INSTRUMENT_BLOCKING_API_HPP


#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>

#include <InstrumentTaskId.hpp>


namespace Instrument {
	//! This function is called upon entering the blocking API task block function
	//! If taskRuntimeTransistion is true, Task Hardware Counters have been updated before calling this function.
	//! \param[in] taskID The blocking task identifier
	//! \param[in] taskRuntimeTransition whether this API function was called from task or
	//! runtime code
	void enterBlockCurrentTask(task_id_t taskId, bool taskRuntimeTransition, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());

	//! This function is called upon exiting blocking API task block function
	//! If taskRuntimeTransistion is true, Runtime Hardware Counters have been updated before calling this function.
	//! \param[in] taskID The blocking task identifier
	//! \param[in] taskRuntimeTransition Whether this API function was called from task or
	//! runtime code
	void exitBlockCurrentTask(task_id_t taskId, bool taskRuntimeTransition, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());

	//! This function is called upon entering the blocking API task unblock function
	//! If taskRuntimeTransistion is true, Task Hardware Counters have been updated before calling this function.
	//! \param[in] taskID The blocking task identifier
	//! \param[in] taskRuntimeTransition Whether this API function was called from task or
	//! runtime code
	void enterUnblockTask(task_id_t taskId, bool taskRuntimeTransition, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());

	//! This function is called upon exiting the blocking API task unblock function
	//! If taskRuntimeTransistion is true, Runtime Hardware Counters have been updated before calling this function.
	//! \param[in] taskID The blocking task identifier
	//! \param[in] taskRuntimeTransition Whether this API function was called from task or
	//! runtime code
	void exitUnblockTask(task_id_t taskId, bool taskRuntimeTransition, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());

	//! This function is called upon entering the blocking API wait for function
	//! Task Hardware Counters are always updated before calling this function
	//! \param[in] taskID The blocking task identifier
	void enterWaitFor(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());

	//! This function is called upon exiting the blocking API wait for function
	//! Runtime Hardware Counters are always updated before calling this function
	//! \param[in] taskID The blocking task identifier
	void exitWaitFor(task_id_t taskid, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
}


#endif // INSTRUMENT_BLOCKING_API_HPP
