/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


#pragma GCC visibility push(default)

void nanos6_register_taskloop_bounds(
	void *task,
	size_t lower_bound,
	size_t upper_bound,
	size_t step,
	size_t chunksize
) {
	typedef void nanos6_register_taskloop_bounds_t(
			void *task, size_t lower_bound, size_t upper_bound, size_t step, size_t chunksize
	);
	
	static nanos6_register_taskloop_bounds_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_register_taskloop_bounds_t *) _nanos6_resolve_symbol("nanos6_register_taskloop_bounds", "essential", NULL);
	}
	
	(*symbol)(task, lower_bound, upper_bound, step, chunksize);
}

#pragma GCC visibility pop

