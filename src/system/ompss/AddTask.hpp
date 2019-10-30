/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef ADD_TASK_HPP
#define ADD_TASK_HPP

#include <nanos6.h>

#include <InstrumentTaskId.hpp>


//! \brief Gets a preallocated task and resets all its values
//! 
//! This function gets a preallocated task and reinitializes all its values.
//! This can only be used as a collaborator of a task for.
//! 
//! This method is not in the public API because it cannot be called from outside the runtime library 
//! 
//! \param[in] task_info a pointer to the nanos6_task_info_t structure
//! \param[in] task_invocation_info a pointer to the nanos6_task_invocation_info_t structure
//! \param[in] parentTaskInstrumentationId instrumentation info from the parent 
//! \param[in] args_block_size size needed to store the parameters passed to the task call
//! \param[in] preallocated_args_block a pointer to a location to store the pointer to the block of data that will contain the parameters of the task call
//! \param[in] preallocated_task a pointer to the preallocated task to be reset
void nanos6_create_preallocated_task(
	nanos6_task_info_t *task_info,
	nanos6_task_invocation_info_t *task_invocation_info,
	Instrument::task_id_t parentTaskInstrumentationId,
	size_t args_block_size,
	void *preallocated_args_block,
	void *preallocated_task,
	size_t flags
);

#endif // ADD_TASK_HPP
