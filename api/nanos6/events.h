/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_EVENTS_H
#define NANOS6_EVENTS_H


#include "major.h"


#pragma GCC visibility push(default)


// NOTE: The full version depends also on nanos6_major_api
//       That is:   nanos6_major_api . nanos6_events_api
enum nanos6_events_api_t { nanos6_events_api = 1 };


#ifdef __cplusplus
extern "C" {
#endif

//! \brief Get the event counter associated with the current task
//!
//! \returns the associated event counter with the executing task
//!
void *nanos6_get_current_event_counter(void);

//! \brief Increase the counter of events of the current task to prevent the release of dependencies
//!
//! This function atomically increases the counter of events of a task
//!
//! \param[in] event_counter The event counter according with the current task
//! \param[in] value The value to be incremented (must be positive or zero)
void nanos6_increase_current_task_event_counter(void *event_counter, unsigned int increment);

//! \brief Decrease the counter of events of a task and release the dependencies if required
//!
//! This function atomically decreases the counter of events of a task and
//! it releases the depencencies once the number of events becomes zero
//! and the task has completed its execution
//!
//! \param[in] event_counter The event counter of the task
//! \param[in] value The value to be decremented (must be positive or zero)
void nanos6_decrease_task_event_counter(void *event_counter, unsigned int decrement);

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif // NANOS6_EVENTS_H
