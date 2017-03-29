#include "resolve.h"


RESOLVE_API_FUNCTION(nanos_get_current_blocking_context, "task blocking", NULL);
RESOLVE_API_FUNCTION(nanos_block_current_task, "task blocking", NULL);
RESOLVE_API_FUNCTION(nanos_unblock_task, "task blocking", NULL);

