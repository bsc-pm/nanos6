/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


#pragma GCC visibility push(default)

void *nanos_get_reduction_storage(const void *address)
{
	typedef void *nanos_get_reduction_storage_t(const void *address);
	
	static nanos_get_reduction_storage_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_get_reduction_storage_t *)
			_nanos6_resolve_symbol("nanos_get_reduction_storage",
					"reductions", NULL);
	}
	
	(*symbol)(address);
}

#pragma GCC visibility pop
