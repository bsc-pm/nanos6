/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef ADD_TASK_HPP
#define ADD_TASK_HPP

#include <nanos6.h>

#include "tasks/Task.hpp"


namespace AddTask {

	//! \brief Allocate space for a task and its arguments
	//!
	//! This function creates a task and allocates space for its parameters. After calling it,
	//! the user should fill out the block of data stored in the task's args block and call
	//! AddTask::submitTask passing the task pointer returned by this function
	//!
	//! \param[in] taskInfo A pointer to the nanos6_task_info_t structure
	//! \param[in] taskInvocationInfo A pointer to the nanos6_task_invocation_info_t structure
	//! \param[in] argsBlock A location to store the data block that will contain the arguments
	//!            of the task call. This parameter is only considered when that data block is
	//!            is preallocated. In any case, the pointer to the args block can be retrieved
	//!            querying the task returned by this function
	//! \param[in] argsBlockSize The size needed to store the parameters passed to the task call
	//! \param[in] flags The flags of the task
	//! \param[in] numDependencies The expected number of task dependencies or -1 if undefined
	//! \param[in] fromUserCode Whether called from user code (i.e. nanos6_create_task)
	//!
	//! \returns The created task
	Task *createTask(
		nanos6_task_info_t *taskInfo,
		nanos6_task_invocation_info_t *taskInvocationInfo,
		void *argsBlock,
		size_t argsBlockSize,
		size_t flags,
		size_t numDependencies = 0,
		bool fromUserCode = false
	);

	//! \brief Submit a task
	//!
	//! This function should be called after filling out the block of parameters of
	//! the task. See the AddTask::createTask function
	//!
	//! \param[in] task The task handler
	//! \param[in] parent The parent task of the submitted task. Note that a task can
	//!            have a creator but may not be considered the parent. This parameter
	//!            can be nullptr
	//! \param[in] fromUserCode Whether called from user code (i.e. nanos6_submit_task)
	void submitTask(Task *task, Task *parent, bool fromUserCode = false);
}

#endif // ADD_TASK_HPP
