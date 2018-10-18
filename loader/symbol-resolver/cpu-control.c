/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


RESOLVE_API_FUNCTION(nanos6_get_current_system_cpu, "cpu control", NULL);
RESOLVE_API_FUNCTION(nanos6_get_current_virtual_cpu, "cpu control", NULL);
RESOLVE_API_FUNCTION(nanos6_enable_cpu, "cpu control", NULL);
RESOLVE_API_FUNCTION(nanos6_disable_cpu, "cpu control", NULL);
RESOLVE_API_FUNCTION(nanos6_get_cpu_status, "cpu control", NULL);
RESOLVE_API_FUNCTION(nanos6_cpus_begin, "cpu control", NULL);
RESOLVE_API_FUNCTION(nanos6_cpus_end, "cpu control", NULL);
RESOLVE_API_FUNCTION(nanos6_cpus_advance, "cpu control", NULL);
RESOLVE_API_FUNCTION(nanos6_cpus_get, "cpu control", NULL);
RESOLVE_API_FUNCTION(nanos6_cpus_get_virtual, "cpu control", NULL);

