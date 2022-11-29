/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_ADD_TASK_HPP
#define INSTRUMENT_ADD_TASK_HPP


#include <nanos6.h>

#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>


class Task;


namespace Instrument {
	//! This function is called upon entering the task creation function and must
	//! return an instrumentation-specific task identifier.
	//! The other 2 functions will also be called by the same thread sequentially.
	//! If taskRuntimeTransistion is true, Task Hardware Counters have been updated before calling this function.
	//! \param[in] taskInfo The taskInfo describing the task
	//! \param[in] taskInvokationInfo A pointer to the nanos6_task_invocation_info_t structure
	//! \param[in] flags The flags of the task
	//! \param[in] taskRuntimeTransition whether this API function was called from task or
	//! runtime code
	task_id_t enterCreateTask(nanos6_task_info_t *taskInfo, nanos6_task_invocation_info_t *taskInvokationInfo, size_t flags, bool taskRuntimeTransition, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());

	//! This function is called upon exiting the task creation function
	//! \param[in] taskRuntimeTransition whether this API function was called from task or
	//! runtime code
	void exitCreateTask(bool taskRuntimeTransition);

	//! This function is called after having created the Task object and before the
	//! task can be executed
	//! \param[in] task the Task object
	//! \param[in] taskId the task identifier returned in the call to enterAddTask
	void createdTask(void *task, task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());

	//! This function is called after having allocated the args block and before
	//! the actual Task object is created.
	//! \param[in] taskId the task identifier returned in the call to enterAddTask
	//! \param[in] argsBlockPointer a pointer to the args block
	//! \param[in] originalArgsBlockSize the original size of the args block,
	//! before any alignment correction
	//! \param[in] argsBlockSize the actual size of the args block, if alignment
	//! was corrected
	void createdArgsBlock(task_id_t taskId, void *argsBlockPointer, size_t originalArgsBlockSize, size_t argsBlockSize, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());

	//! This function is called upon entering the submit task function
	//! \param[in] taskRuntimeTransition whether this API was called from task (true) or
	//! runtime (false) code
	void enterSubmitTask(bool taskRuntimeTransition);

	//! This function is called upon exiting the submit task function. The task
	//! identifier is necessary because the actual task may have already been
	//! destroyed by the time this function is called
	//! If taskRuntimeTransistion is true, Runtime Hardware Counters have been updated before calling this function.
	//! \param[in] taskId the task identifier returned in the call to enterAddTask
	//! \param[in] taskRuntimeTransition whether this API was called from task or
	//! runtime code
	void exitSubmitTask(task_id_t taskId, bool taskRuntimeTransition, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());

	//! This function is called within Nanos6 core, just after registering a
	//! a new spawned task type but before creating the task.
	//! \param[in] the task info describing the task
	void registeredNewSpawnedTaskType(nanos6_task_info_t *taskInfo);

	//! This function is called upon entering SpawnFunction::spawnFunction
	//! If taskRuntimeTransistion is true, Task Hardware Counters have been updated before calling this function.
	//! \param[in] taskRuntimeTransition whether this API function was called from task or
	//! runtime code
	void enterSpawnFunction(bool taskRuntimeTransition);

	//! This function is called upon exiting SpawnFunction::spawnFunction
	//! If taskRuntimeTransistion is true, Runtime Hardware Counters have been updated before calling this function.
	//! \param[in] taskRuntimeTransition whether this API function was called from task or
	//! runtime code
	void exitSpawnFunction(bool taskRuntimeTransition);
}


#endif // INSTRUMENT_ADD_TASK_HPP
