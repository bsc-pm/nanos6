/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


#pragma GCC visibility push(default)

void nanos6_register_task_info(nanos6_task_info_t *task_info)
{
	typedef void nanos6_register_task_info_t(nanos6_task_info_t *task_info);

	static nanos6_register_task_info_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_register_task_info_t *) _nanos6_resolve_symbol("nanos6_register_task_info", "essential", NULL);
	}

	(*symbol)(task_info);
}

#pragma GCC visibility pop

