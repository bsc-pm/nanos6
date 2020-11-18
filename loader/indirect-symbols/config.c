/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


#pragma GCC visibility push(default)

void nanos6_config_assert(const char *config_conditions)
{
	typedef void nanos6_config_assert_t(const char *config_conditions);

	static nanos6_config_assert_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_config_assert_t *) _nanos6_resolve_symbol("nanos6_config_assert", "essential", NULL);
	}

	(*symbol)(config_conditions);
}

#pragma GCC visibility pop

