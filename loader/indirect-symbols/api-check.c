/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"
#include <api/nanos6/api-check.h>


#pragma GCC visibility push(default)

int nanos6_check_api_versions(nanos6_api_versions_t const *api_versions)
{
	typedef int nanos6_check_api_versions_t(nanos6_api_versions_t const *api_versions);
	
	static nanos6_check_api_versions_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_check_api_versions_t *) _nanos6_resolve_symbol("nanos6_check_api_versions", "API versioning", NULL);
		if (symbol == NULL) {
			return 0;
		}
	}
	
	return (*symbol)(api_versions);
}

#pragma GCC visibility pop
