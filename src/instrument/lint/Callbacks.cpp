/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "Callbacks.hpp"


#ifdef __cplusplus
extern "C" {
#endif


void nanos6_lint_on_task_creation(
	__attribute__((unused)) long task_id,
	__attribute__((unused)) nanos6_task_invocation_info_t *taskInvokationInfo,
	__attribute__((unused)) size_t flags
)
{
	return;
}


void nanos6_lint_on_task_argsblock_allocation(
	__attribute__((unused)) long task_id,
	__attribute__((unused)) void *args_block_ptr,
	__attribute__((unused)) size_t orig_args_block_size,
	__attribute__((unused)) size_t args_block_size
)
{
	return;
}


void nanos6_lint_on_task_start(
	__attribute__((unused)) long task_id
)
{
	return;
}


void nanos6_lint_on_task_end(
	__attribute__((unused)) long task_id
)
{
	return;
}


void nanos6_lint_on_task_destruction(
	__attribute__((unused)) long task_id
)
{
	return;
}


void nanos6_lint_on_taskwait_enter(
	__attribute__((unused)) long task_id,
	__attribute__((unused)) char const *invocationSource,
	__attribute__((unused)) long if0TaskId
)
{
	return;
}


void nanos6_lint_on_taskwait_exit(
	__attribute__((unused)) long task_id
)
{
	return;
}


#ifdef __cplusplus
}
#endif
