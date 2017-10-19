/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


#pragma GCC visibility push(default)

void nanos_register_taskloop_boundsinfo1(
		void *handler,
		char const *it1_text, size_t it1_lower_bound, size_t it1_upper_bound, size_t it1_step_size, size_t it1_grid_size
) {
	typedef void nanos_register_taskloop_boundsinfo1_t(void *handler,
		char const *it1_text, size_t it1_lower_bound, size_t it1_upper_bound, size_t it1_step_size, size_t it1_grid_size
	);
	
	static nanos_register_taskloop_boundsinfo1_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_register_taskloop_boundsinfo1_t *) _nanos6_resolve_symbol("nanos_register_taskloop_boundsinfo1", "essential", NULL);
	}
	
	(*symbol)(handler, it1_text, it1_lower_bound, it1_upper_bound, it1_step_size, it1_grid_size);
}

void nanos_register_taskloop_boundsinfo2(
		void *handler,
		char const *it1_text, size_t it1_lower_bound, size_t it1_upper_bound, size_t it1_step_size, size_t it1_grid_size,
		char const *it2_text, size_t it2_lower_bound, size_t it2_upper_bound, size_t it2_step_size, size_t it2_grid_size
) {
	typedef void nanos_register_taskloop_boundsinfo2_t(void *handler,
		char const *it1_text, size_t it1_lower_bound, size_t it1_upper_bound, size_t it1_step_size, size_t it1_grid_size,
		char const *it2_text, size_t it2_lower_bound, size_t it2_upper_bound, size_t it2_step_size, size_t it2_grid_size
	);
	
	static nanos_register_taskloop_boundsinfo2_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_register_taskloop_boundsinfo2_t *) _nanos6_resolve_symbol("nanos_register_taskloop_boundsinfo2", "essential", NULL);
	}
	
	(*symbol)(handler,
			it1_text, it1_lower_bound, it1_upper_bound, it1_step_size, it1_grid_size,
			it2_text, it2_lower_bound, it2_upper_bound, it2_step_size, it2_grid_size
	);
}

#pragma GCC visibility pop

