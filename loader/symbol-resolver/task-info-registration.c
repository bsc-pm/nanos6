/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


static void nanos6_register_task_info_unused(__attribute__((unused)) void *task_info)
{
}


RESOLVE_API_FUNCTION_WITH_LOCAL_FALLBACK(nanos6_register_task_info, "essential", nanos6_register_task_info_unused);
