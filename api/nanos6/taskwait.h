/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_TASKWAIT_H
#define NANOS6_TASKWAIT_H

#include "major.h"


#pragma GCC visibility push(default)


// NOTE: The full version depends also on nanos6_major_api
//       That is:   nanos6_major_api . nanos6_taskwait_api
enum nanos6_taskwait_api_t { nanos6_taskwait_api = 3 };


#ifdef __cplusplus
extern "C" {
#endif


//! \brief Block the control flow of the current task until all of its children have finished
//!
//! \param[in] invocation_source A string that identifies the source code location of the invocation
void nanos6_taskwait(char const *invocation_source);

//! \brief Block the control flow in a stream until all previously spawned
//! functions have finished
//! \param[in] stream_id The identifier of the stream to synchronize
void nanos6_stream_synchronize(size_t stream_id);

//! \brief Block the control flow until all previously spawned functions in
//! all the existing streams have finished
void nanos6_stream_synchronize_all(void);

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif // NANOS6_TASKWAIT_H
