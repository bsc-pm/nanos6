/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
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

	//! This function is called after having allocated the args block and before
	//! the actual Task object is created.
	//! \param[in] taskId the task identifier returned in the call to enterAddTask
	//! \param[in] argsBlockPointer a pointer to the args block
	//! \param[in] originalArgsBlockSize the original size of the args block,
	//!            before any alignment correction
	//! \param[in] argsBlockSize the actual size of the args block, if alignment
	//!            was corrected
	void createdArgsBlock(task_id_t taskId, void *argsBlockPointer, size_t originalArgsBlockSize, size_t argsBlockSize, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	
	//! This function is called right before returning to the user code. The task
	//! identifier is necessary because the actual task may have already been
	//! destroyed by the time this function is called.
	//! \param[in] taskId the task identifier returned in the call to enterAddTask
	void exitAddTask(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	
	//! This function is called right after entering the runtime and must
	//! return an instrumentation-specific task identifier.
	//! The other 2 functions will also be called by the same thread sequentially.
	//! \param[in] taskforId the task identifier of the source taskfor returned in the call to enterAddTask
	//! \param[in] collaboratorId the task identifier of the collaborator returned in the call to enterAddTask
	task_id_t enterAddTaskforCollaborator(task_id_t taskforId, nanos6_task_info_t *taskInfo, nanos6_task_invocation_info_t *taskInvokationInfo, size_t flags, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	
	//! This function is called right before returning to the user code. The task
	//! identifier is necessary because the actual task may have already been
	//! destroyed by the time this function is called.
	//! \param[in] taskforId the task identifier of the source taskfor returned in the call to enterAddTask
	//! \param[in] collaboratorId the task identifier of the collaborator returned in the call to enterAddTask
	void exitAddTaskforCollaborator(task_id_t taskforId, task_id_t collaboratorId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());

	//! This function is called within Nanos6 core, just after registering a
	//! a new spawned task type but before creating the task.
	//! \param[in] the task info describing the task
	void registeredNewSpawnedTaskType(nanos6_task_info_t *taskInfo);
}


#endif // INSTRUMENT_ADD_TASK_HPP
