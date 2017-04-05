#include "resolve.h"


static void nanos_register_task_info_unused(__attribute__((unused)) void *task_info)
{
}


RESOLVE_API_FUNCTION_WITH_LOCAL_FALLBACK(nanos_register_task_info, "essential", nanos_register_task_info_unused);
