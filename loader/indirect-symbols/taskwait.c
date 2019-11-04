/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
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

void nanos6_stream_synchronize(size_t stream_id)
{
	typedef void nanos6_stream_synchronize_t(size_t stream_id);

	static nanos6_stream_synchronize_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_stream_synchronize_t *) _nanos6_resolve_symbol("nanos6_stream_synchronize", "essential", NULL);
	}

	(*symbol)(stream_id);
}

void nanos6_stream_synchronize_all(void)
{
	typedef void nanos6_stream_synchronize_all_t(void);

	static nanos6_stream_synchronize_all_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_stream_synchronize_all_t *) _nanos6_resolve_symbol("nanos6_stream_synchronize_all", "essential", NULL);
	}

	(*symbol)();
}

#pragma GCC visibility pop

