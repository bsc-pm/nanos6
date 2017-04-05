#include "resolve.h"


static signed int signed_int_always_false(void) { return 0; }
RESOLVE_API_FUNCTION_WITH_LOCAL_FALLBACK(nanos_in_final, "final tasks", signed_int_always_false);

