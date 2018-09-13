/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


static signed int signed_int_always_false(void) { return 0; }
RESOLVE_API_FUNCTION_WITH_LOCAL_FALLBACK(nanos6_in_final, "final tasks", signed_int_always_false);
RESOLVE_API_FUNCTION_WITH_LOCAL_FALLBACK(nanos6_in_serial_context, "final tasks", signed_int_always_false);

