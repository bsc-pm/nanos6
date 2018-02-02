/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_TASK_STATUS_HPP
#define INSTRUMENT_TASK_STATUS_HPP


#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>


namespace Instrument {
	//! \brief This type here is for convenience only
	typedef enum {
		//! \brief The task is being created
		in_creation_task_status,
		
		//! \brief The task has predecessors and thus is still not ready
		pending_task_status,
		
		//! \brief The task is ready for execution
		ready_task_status,
		
		//! \brief The task is being executed
		executing_task_status,
		
		//! \brief The task is blocked, the reason is detailed with a task_blocking_reason_t value
		blocked_task_status,
		
		//! \brief The task is zoombie, i.e. it has finished but it has not been deleted yet
		zombie_task_status
	} task_status_t;
	
	
	//! \brief The reason for a task getting blocked during its execution
	typedef enum {
		in_taskwait_blocking_reason,
		in_mutex_blocking_reason,
		user_requested_blocking_reason
	} task_blocking_reason_t;
	
	//! NOTE: Instrument::createdTask gets called before any of the following functions and before calculating the dependencies
	
	//! \brief Indicates that the task is currently pending, i.e. it has predecessors
	//! \param[in] taskId the task identifier returned in the call to enterAddTask
	void taskIsPending(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	
	//! \brief Indicates that the task is currently ready to be executed
	//! \param[in] taskId the task identifier returned in the call to enterAddTask
	void taskIsReady(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	
	//! \brief Indicates that the task is currently being executed
	//! \param[in] taskId the task identifier returned in the call to enterAddTask
	void taskIsExecuting(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	
	//! \brief Indicates that the task is currently blocked
	//! 
	//! The task may exit this state by becoming ready (sent back to the scheduler), or executing
	//! 
	//! \param[in] taskId the task identifier returned in the call to enterAddTask
	//! \param[in] reason the reason why the task gets blocked
	void taskIsBlocked(task_id_t taskId, task_blocking_reason_t reason, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	
	//! \brief Indicates that the task has finished and at some point will be destroyed
	//! \param[in] taskId the task identifier returned in the call to enterAddTask
	void taskIsZombie(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	
	//! \brief Indicates that the task is about to be destroyed
	//! \param[in] taskId the task identifier returned in the call to enterAddTask
	void taskIsBeingDeleted(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	
	void taskHasNewPriority(task_id_t taskId, long priority, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
}


#endif // INSTRUMENT_TASK_STATUS_HPP
