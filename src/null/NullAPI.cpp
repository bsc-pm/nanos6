// This is for posix_memalign
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include "api/nanos6_rt_interface.h"


void nanos_taskwait(__attribute__((unused)) char const *invocationSource)
{
}


void nanos_user_lock(__attribute__((unused)) void **handlerPointer, __attribute__((unused)) char const *invocationSource)
{
}


void nanos_user_unlock(__attribute__((unused)) void **handlerPointer)
{
}

