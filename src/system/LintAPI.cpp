/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/lint.h>



extern "C" void nanos6_lint_ignore_region_begin(void)
{
	return;
}


extern "C" void nanos6_lint_ignore_region_end(void)
{
	return;
}


extern "C" void nanos6_lint_register_alloc(
	__attribute__((unused)) void *base_address,
	__attribute__((unused)) unsigned long size
)
{
	return;
}


extern "C" void nanos6_lint_register_free(
	__attribute__((unused)) void *base_address
)
{
	return;
}


