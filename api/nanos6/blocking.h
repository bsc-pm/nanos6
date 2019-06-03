/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_BLOCKING_H
#define NANOS6_BLOCKING_H

#include "major.h"


#pragma GCC visibility push(default)


// NOTE: The full version depends also on nanos6_major_api
//       That is:   nanos6_major_api . nanos6_blocking_api
enum nanos6_blocking_api_t { nanos6_blocking_api = 1 };

#ifdef __cplusplus
extern "C" {
#endif


//! \brief Get a handler that allows to block and unblock the current task
//!
//! \returns an opaque pointer that is used for blocking and unblocking
//! the current task.
//! 
//! The underlying implementation may or may not return the same value
//! for repeated calls to this function.
//! 
//! Once the handler has been used once in a call to nanos6_block_current_task
//! and a call to nanos6_unblock_task, the handler is discarded and a new
//! one must be obtained to perform another cycle of blocking and unblocking.
void *nanos6_get_current_blocking_context(void);

//! \brief Block the current task
//! 
//! \param blocking_context the value returned by a call to nanos6_get_current_blocking_context
//! performed within the task
//! 
//! This function blocks the execution of the current task at least until
//! a thread calls nanos6_unblock_task with its blocking context
//! 
//! The runtime may choose to execute other tasks within the execution scope
//! of this call.
void nanos6_block_current_task(void *blocking_context);

//! \brief Unblock a task previously or about to be blocked
//! 
//! Mark as unblocked a task previously or about to be blocked inside a call
//! to nanos6_block_current_task.
//! 
//! While this function can be called before the actual to nanos6_block_current_task,
//! only one call to it may precede its matching call to nanos6_block_current_task.
//! 
//! The return of this function does not guarantee that the blocked task has
//! resumed yet its execution. It only guarantees that it will be resumed.
//! 
//! \param[in] blocking_context the handler used to block the task
void nanos6_unblock_task(void *blocking_context);


#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop

#endif /* NANOS6_BLOCKING_H */
