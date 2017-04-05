#include "resolve.h"


RESOLVE_API_FUNCTION(nanos_register_read_depinfo, "dependency", NULL);
RESOLVE_API_FUNCTION(nanos_register_write_depinfo, "dependency", NULL);
RESOLVE_API_FUNCTION(nanos_register_readwrite_depinfo, "dependency", NULL);

RESOLVE_API_FUNCTION(nanos_register_commutative_depinfo, "commutative dependency", nanos_register_readwrite_depinfo);
RESOLVE_API_FUNCTION(nanos_register_concurrent_depinfo, "concurrent dependency", nanos_register_readwrite_depinfo);

