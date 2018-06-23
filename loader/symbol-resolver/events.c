/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


RESOLVE_API_FUNCTION(nanos_get_current_event_counter, "essential", NULL);
RESOLVE_API_FUNCTION(nanos_increase_current_task_event_counter, "essential", NULL);
RESOLVE_API_FUNCTION(nanos_decrease_task_event_counter, "essential", NULL);
