/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


#pragma GCC visibility push(default)

void nanos6_taskwait(char const *invocation_source)
{
	typedef void nanos6_taskwait_t(char const *invocation_source);
	
	static nanos6_taskwait_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_taskwait_t *) _nanos6_resolve_symbol("nanos6_taskwait", "essential", NULL);
	}
	
	(*symbol)(invocation_source);
}

#pragma GCC visibility pop

