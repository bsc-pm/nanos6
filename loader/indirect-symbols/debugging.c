/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"
#include "api/nanos6/debug.h"


#pragma GCC visibility push(default)

char const *nanos6_get_runtime_version(void)
{
	typedef char const *nanos6_get_runtime_version_t(void);
	
	static nanos6_get_runtime_version_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_get_runtime_version_t *) _nanos6_resolve_symbol("nanos6_get_runtime_version", "debugging", NULL);
	}
	
	return (*symbol)();
}


char const *nanos6_get_runtime_copyright(void)
{
	typedef char const *nanos6_get_runtime_copyright_t(void);
	
	static nanos6_get_runtime_copyright_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_get_runtime_copyright_t *) _nanos6_resolve_symbol("nanos6_get_runtime_copyright", "licensing", NULL);
	}
	
	return (*symbol)();
}


char const *nanos6_get_runtime_license(void)
{
	typedef char const *nanos6_get_runtime_license_t(void);
	
	static nanos6_get_runtime_license_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_get_runtime_license_t *) _nanos6_resolve_symbol("nanos6_get_runtime_license", "licensing", NULL);
	}
	
	return (*symbol)();
}


char const *nanos6_get_runtime_full_license(void)
{
	typedef char const *nanos6_get_runtime_full_license_t(void);
	
	static nanos6_get_runtime_full_license_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_get_runtime_full_license_t *) _nanos6_resolve_symbol("nanos6_get_runtime_full_license", "licensing", NULL);
	}
	
	return (*symbol)();
}


char const *nanos6_get_runtime_branch(void)
{
	typedef char const *nanos6_get_runtime_branch_t(void);
	
	static nanos6_get_runtime_branch_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_get_runtime_branch_t *) _nanos6_resolve_symbol("nanos6_get_runtime_branch", "debugging", NULL);
	}
	
	return (*symbol)();
}


char const *nanos6_get_runtime_patches(void)
{
	typedef char const *nanos6_get_runtime_patches_t(void);
	
	static nanos6_get_runtime_patches_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_get_runtime_patches_t *) _nanos6_resolve_symbol("nanos6_get_runtime_patches", "debugging", NULL);
	}
	
	return (*symbol)();
}


char const *nanos6_get_runtime_compiler_version(void)
{
	typedef char const *nanos6_get_runtime_compiler_version_t(void);
	
	static nanos6_get_runtime_compiler_version_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_get_runtime_compiler_version_t *) _nanos6_resolve_symbol("nanos6_get_runtime_compiler_version", "debugging", NULL);
	}
	
	return (*symbol)();
}


char const *nanos6_get_runtime_compiler_flags(void)
{
	typedef char const *nanos6_get_runtime_compiler_flags_t(void);
	
	static nanos6_get_runtime_compiler_flags_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_get_runtime_compiler_flags_t *) _nanos6_resolve_symbol("nanos6_get_runtime_compiler_flags", "debugging", NULL);
	}
	
	return (*symbol)();
}



void nanos6_wait_for_full_initialization(void)
{
	typedef void nanos6_wait_for_full_initialization_t(void);
	
	static nanos6_wait_for_full_initialization_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_wait_for_full_initialization_t *) _nanos6_resolve_symbol("nanos6_wait_for_full_initialization", "debugging", NULL);
	}
	
	(*symbol)();
}


unsigned int nanos6_get_num_cpus(void)
{
	typedef unsigned int nanos6_get_num_cpus_t(void);
	
	static nanos6_get_num_cpus_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_get_num_cpus_t *) _nanos6_resolve_symbol("nanos6_get_num_cpus", "debugging", NULL);
	}
	
	return (*symbol)();
}


#pragma GCC visibility pop
