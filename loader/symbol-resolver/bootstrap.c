/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


RESOLVE_API_FUNCTION(nanos6_preinit, "essential", NULL);
RESOLVE_API_FUNCTION(nanos6_can_run_main, "essential", NULL);
RESOLVE_API_FUNCTION(nanos6_register_completion_callback, "essential", NULL);
RESOLVE_API_FUNCTION(nanos6_init, "essential", NULL);
RESOLVE_API_FUNCTION(nanos6_shutdown, "essential", NULL);

