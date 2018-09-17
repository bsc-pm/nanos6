/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_ADD_TASK_HPP
#define INSTRUMENT_ADD_TASK_HPP


#include <nanos6.h>

#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>


class Task;


namespace Instrument {
	//! This function is called right after entering the runtime and must
	//! return an instrumentation-specific task identifier.
	//! The other 2 functions will also be called by the same thread sequentially.
	task_id_t enterAddTask(nanos6_task_info_t *taskInfo, nanos6_task_invocation_info_t *taskInvokationInfo, size_t flags, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	
	//! This function is called after having created the Task object and before the
	//! task can be executed.
	//! \param[in] task the Task object
	//! \param[in] taskId the task identifier returned in the call to enterAddTask
	void createdTask(void *task, task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	
	//! This function is called right before returning to the user code. The task
	//! identifier is necessary because the actual task may have already been
	//! destroyed by the time this function is called.
	//! \param[in] taskId the task identifier returned in the call to enterAddTask
	void exitAddTask(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
}


#endif // INSTRUMENT_ADD_TASK_HPP
