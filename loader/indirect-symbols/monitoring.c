/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"
#include "api/nanos6/monitoring.h"


#pragma GCC visibility push(default)

double nanos6_get_predicted_elapsed_time(void)
{
	typedef double nanos6_get_predicted_elapsed_time_t(void);
	
	static nanos6_get_predicted_elapsed_time_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_get_predicted_elapsed_time_t *) _nanos6_resolve_symbol("nanos6_get_predicted_elapsed_time", "monitoring", NULL);
	}
	
	return (*symbol)();
}

#pragma GCC visibility pop
