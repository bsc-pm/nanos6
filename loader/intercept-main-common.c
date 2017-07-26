/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <assert.h>
#include <dlfcn.h>
#include <stddef.h>


extern void *_nanos6_loader_next_libc_start_main;


__attribute__ ((visibility ("hidden"))) void _nanos6_resolve_next_start_main(char const *name)
{
	// Retrieve the next __libc_start_main which problably will be the one in libc
	_nanos6_loader_next_libc_start_main = dlsym(RTLD_NEXT, name);
	assert(_nanos6_loader_next_libc_start_main != NULL);
}

