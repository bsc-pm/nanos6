/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/utils.h>

#include <string.h>


void nanos6_bzero(void *buffer, size_t size)
{
	memset(buffer, 0, size);
}

