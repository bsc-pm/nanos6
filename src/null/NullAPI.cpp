// This is for posix_memalign
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include <nanos6.h>


void nanos_taskwait(__attribute__((unused)) char const *invocationSource)
{
}


void nanos_user_lock(__attribute__((unused)) void **handlerPointer, __attribute__((unused)) char const *invocationSource)
{
}


void nanos_user_unlock(__attribute__((unused)) void **handlerPointer)
{
}


void nanos_preinit(void)
{
}

void nanos_init(void)
{
}

void nanos_wait_until_shutdown(void)
{
}

void nanos_notify_ready_for_shutdown(void)
{
}
