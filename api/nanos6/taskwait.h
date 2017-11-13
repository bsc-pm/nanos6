/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_TASKWAIT_H
#define NANOS6_TASKWAIT_H


#pragma GCC visibility push(default)

enum nanos6_taskwait_api_t { nanos6_taskwait_api = 1 };


#ifdef __cplusplus
extern "C" {
#endif


//! \brief Block the control flow of the current task until all of its children have finished
//!
//! \param[in] invocation_source A string that identifies the source code location of the invocation
void nanos_taskwait(char const *invocation_source);


#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif // NANOS6_TASKWAIT_H
