#include "resolve.h"
#include "api/nanos6/debug.h"


char const *nanos_get_runtime_version(void)
{
	typedef char const *nanos_get_runtime_version_t();
	
	static nanos_get_runtime_version_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_get_runtime_version_t *) _nanos6_resolve_symbol("nanos_get_runtime_version", "debugging", NULL);
	}
	
	return (*symbol)();
}


char const *nanos_get_runtime_branch(void)
{
	typedef char const *nanos_get_runtime_branch_t();
	
	static nanos_get_runtime_branch_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_get_runtime_branch_t *) _nanos6_resolve_symbol("nanos_get_runtime_branch", "debugging", NULL);
	}
	
	return (*symbol)();
}


char const *nanos_get_runtime_patches(void)
{
	typedef char const *nanos_get_runtime_patches_t();
	
	static nanos_get_runtime_patches_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_get_runtime_patches_t *) _nanos6_resolve_symbol("nanos_get_runtime_patches", "debugging", NULL);
	}
	
	return (*symbol)();
}


char const *nanos_get_runtime_compiler_version(void)
{
	typedef char const *nanos_get_runtime_compiler_version_t();
	
	static nanos_get_runtime_compiler_version_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_get_runtime_compiler_version_t *) _nanos6_resolve_symbol("nanos_get_runtime_compiler_version", "debugging", NULL);
	}
	
	return (*symbol)();
}


char const *nanos_get_runtime_compiler_flags(void)
{
	typedef char const *nanos_get_runtime_compiler_flags_t();
	
	static nanos_get_runtime_compiler_flags_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_get_runtime_compiler_flags_t *) _nanos6_resolve_symbol("nanos_get_runtime_compiler_flags", "debugging", NULL);
	}
	
	return (*symbol)();
}



void nanos_wait_for_full_initialization(void)
{
	typedef void nanos_wait_for_full_initialization_t();
	
	static nanos_wait_for_full_initialization_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_wait_for_full_initialization_t *) _nanos6_resolve_symbol("nanos_wait_for_full_initialization", "debugging", NULL);
	}
	
	(*symbol)();
}


unsigned int nanos_get_num_cpus(void)
{
	typedef unsigned int nanos_get_num_cpus_t();
	
	static nanos_get_num_cpus_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_get_num_cpus_t *) _nanos6_resolve_symbol("nanos_get_num_cpus", "debugging", NULL);
	}
	
	return (*symbol)();
}

