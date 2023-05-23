/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


#pragma GCC visibility push(default)

void nanos6_check_version(uint64_t size, nanos6_version_t *versions, const char *source)
{
	typedef void nanos6_check_version_t(
		uint64_t size,
		nanos6_version_t *versions,
		const char *source
	);

	static nanos6_check_version_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_check_version_t *) _nanos6_resolve_symbol("nanos6_check_version", "essential", NULL);
	}

	(*symbol)(size, versions, source);
}

#pragma GCC visibility pop
