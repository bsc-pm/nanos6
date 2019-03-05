/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


#pragma GCC visibility push(default)

void nanos6_create_task(
	nanos6_task_info_t *task_info,
	nanos6_task_invocation_info_t *task_invocation_info,
	size_t args_block_size,
	/* OUT */ void **args_block_pointer,
	/* OUT */ void **task_pointer,
	size_t flags,
	size_t num_deps
) {
	typedef void nanos6_create_task_t(
		nanos6_task_info_t *task_info,
		nanos6_task_invocation_info_t *task_invocation_info,
		size_t args_block_size,
		/* OUT */ void **args_block_pointer,
		/* OUT */ void **task_pointer,
		size_t flags,
		size_t num_deps
	);
	
	static nanos6_create_task_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_create_task_t *) _nanos6_resolve_symbol("nanos6_create_task", "essential", NULL);
	}
	
	(*symbol)(task_info, task_invocation_info, args_block_size, args_block_pointer, task_pointer, flags, num_deps);
}


void nanos6_submit_task(void *task)
{
	typedef void nanos6_submit_task_t(void *task);
	
	static nanos6_submit_task_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_submit_task_t *) _nanos6_resolve_symbol("nanos6_submit_task", "essential", NULL);
	}
	
	(*symbol)(task);
}


void nanos6_spawn_function(void (*function)(void *), void *args, void (*completion_callback)(void *), void *completion_args, char const *label)
{
	typedef void nanos6_spawn_function_t(void (*function)(void *), void *args, void (*completion_callback)(void *), void *completion_args, char const *label);
	
	static nanos6_spawn_function_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_spawn_function_t *) _nanos6_resolve_symbol("nanos6_spawn_function", "essential", NULL);
	}
	
	(*symbol)(function, args, completion_callback, completion_args, label);
}

#pragma GCC visibility pop


