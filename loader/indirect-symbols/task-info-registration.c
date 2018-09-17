/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


static void nanos6_register_task_info_unused(__attribute__((unused)) nanos6_task_info_t *task_info)
{
}


#pragma GCC visibility push(default)

void nanos6_register_task_info(nanos6_task_info_t *task_info)
{
	typedef void nanos6_register_task_info_t(nanos6_task_info_t *task_info);
	
	static nanos6_register_task_info_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_register_task_info_t *) _nanos6_resolve_symbol_with_local_fallback("nanos6_register_task_info", "essential", nanos6_register_task_info_unused, "skipping");
	}
	
	(*symbol)(task_info);
}

#pragma GCC visibility pop

