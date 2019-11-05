/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_LINT_CALLBACKS_HPP
#define INSTRUMENT_LINT_CALLBACKS_HPP


#include <nanos6/task-instantiation.h>


#ifdef __cplusplus
extern "C" {
#endif


void nanos6_lint_on_task_creation(
	long task_id,
	nanos6_task_invocation_info_t *taskInvokationInfo,
	size_t flags
);

void nanos6_lint_on_task_argsblock_allocation(
	long task_id,
	void *args_block_ptr,
	size_t orig_args_block_size,
	size_t args_block_size
);

void nanos6_lint_on_task_start(
	long task_id
);

void nanos6_lint_on_task_end(
	long task_id
);

void nanos6_lint_on_task_destruction(
	long task_id
);

void nanos6_lint_on_taskwait_enter(
	long task_id,
	char const *invocationSource,
	long if0TaskId
);

void nanos6_lint_on_taskwait_exit(
	long task_id
);


#ifdef __cplusplus
}
#endif


#endif // INSTRUMENT_LINT_CALLBACKS_HPP
