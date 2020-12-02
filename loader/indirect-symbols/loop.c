/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


#pragma GCC visibility push(default)

void nanos6_create_loop(
	nanos6_task_info_t *task_info,
	nanos6_task_invocation_info_t *task_invocation_info,
	size_t args_block_size,
	void **args_block_pointer,
	void **task_pointer,
	size_t flags,
	size_t num_deps,
	size_t lower_bound,
	size_t upper_bound,
	size_t grainsize,
	size_t chunksize
) {
	typedef void nanos6_create_loop_t(
		nanos6_task_info_t *task_info,
		nanos6_task_invocation_info_t *task_invocation_info,
		size_t args_block_size,
		void **args_block_pointer,
		void **task_pointer,
		size_t flags,
		size_t num_deps,
		size_t lower_bound,
		size_t upper_bound,
		size_t grainsize,
		size_t chunksize
	);


	static nanos6_create_loop_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_create_loop_t *) _nanos6_resolve_symbol("nanos6_create_loop", "essential", NULL);
	}

	(*symbol)(task_info, task_invocation_info, args_block_size,
		args_block_pointer, task_pointer, flags, num_deps,
		lower_bound, upper_bound, grainsize, chunksize);
}

#pragma GCC visibility pop

