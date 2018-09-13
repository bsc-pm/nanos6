/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


#pragma GCC visibility push(default)

void *nanos6_get_reduction_storage1(
	void *original,
	long dim1size, long dim1start, long dim1end)
{
	typedef void *nanos6_get_reduction_storage1_t(void *original, long dim1size, long dim1start, long dim1end);
	
	static nanos6_get_reduction_storage1_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_get_reduction_storage1_t *)
			_nanos6_resolve_symbol("nanos6_get_reduction_storage1",
					"reductions", NULL);
	}
	
	(*symbol)(original, dim1size, dim1start, dim1end);
}

#pragma GCC visibility pop
