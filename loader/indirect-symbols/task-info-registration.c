/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


static void nanos_register_task_info_unused(__attribute__((unused)) nanos_task_info *task_info)
{
}


#pragma GCC visibility push(default)

void nanos_register_task_info(nanos_task_info *task_info)
{
	typedef void nanos_register_task_info_t(nanos_task_info *task_info);
	
	static nanos_register_task_info_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_register_task_info_t *) _nanos6_resolve_symbol_with_local_fallback("nanos_register_task_info", "essential", nanos_register_task_info_unused, "skipping");
	}
	
	(*symbol)(task_info);
}

#pragma GCC visibility pop

