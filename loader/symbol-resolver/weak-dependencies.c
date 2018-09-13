/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


RESOLVE_API_FUNCTION(nanos6_register_weak_read_depinfo, "weak dependency", nanos6_register_read_depinfo);
RESOLVE_API_FUNCTION(nanos6_register_weak_write_depinfo, "weak dependency", nanos6_register_write_depinfo);
RESOLVE_API_FUNCTION(nanos6_register_weak_readwrite_depinfo, "weak dependency", nanos6_register_readwrite_depinfo);

