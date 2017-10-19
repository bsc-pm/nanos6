/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


#pragma GCC visibility push(default)

void nanos6_bzero(void *buffer, size_t size)
{
	typedef void nanos6_bzero_t(void *buffer, size_t size);
	
	static nanos6_bzero_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_bzero_t *) _nanos6_resolve_symbol("nanos6_bzero", "auxiliary functionality", NULL);
	}
	
	(*symbol)(buffer, size);
}

#pragma GCC visibility pop

