/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


RESOLVE_API_FUNCTION(nanos6_register_read_depinfo, "dependency", NULL);
RESOLVE_API_FUNCTION(nanos6_register_write_depinfo, "dependency", NULL);
RESOLVE_API_FUNCTION(nanos6_register_readwrite_depinfo, "dependency", NULL);

RESOLVE_API_FUNCTION(nanos6_register_commutative_depinfo, "commutative dependency", nanos6_register_readwrite_depinfo);
RESOLVE_API_FUNCTION(nanos6_register_concurrent_depinfo, "concurrent dependency", nanos6_register_readwrite_depinfo);

