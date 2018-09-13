/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

// This is for posix_memalign
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include <nanos6.h>


void nanos6_taskwait(__attribute__((unused)) char const *invocationSource)
{
}


void nanos6_user_lock(__attribute__((unused)) void **handlerPointer, __attribute__((unused)) char const *invocationSource)
{
}


void nanos6_user_unlock(__attribute__((unused)) void **handlerPointer)
{
}


void nanos6_preinit(void)
{
}

void nanos6_init(void)
{
}

void nanos6_shutdown(void)
{
}
