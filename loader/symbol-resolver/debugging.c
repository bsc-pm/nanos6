/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


RESOLVE_API_FUNCTION(nanos6_get_runtime_version, "debugging", NULL);
RESOLVE_API_FUNCTION(nanos6_get_runtime_copyright, "licensing", NULL);
RESOLVE_API_FUNCTION(nanos6_get_runtime_license, "licensing", NULL);
RESOLVE_API_FUNCTION(nanos6_get_runtime_full_license, "licensing", NULL);
RESOLVE_API_FUNCTION(nanos6_get_runtime_branch, "debugging", NULL);
RESOLVE_API_FUNCTION(nanos6_get_runtime_patches, "debugging", NULL);
RESOLVE_API_FUNCTION(nanos6_get_runtime_compiler_version, "debugging", NULL);
RESOLVE_API_FUNCTION(nanos6_get_runtime_compiler_flags, "debugging", NULL);

RESOLVE_API_FUNCTION(nanos6_wait_for_full_initialization, "debugging", NULL);

RESOLVE_API_FUNCTION(nanos6_get_num_cpus, "debugging", NULL);

