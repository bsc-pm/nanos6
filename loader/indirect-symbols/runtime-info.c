/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"
#include "api/nanos6/runtime-info.h"


#pragma GCC visibility push(default)

void *nanos6_runtime_info_begin(void)
{
	typedef void *nanos6_runtime_info_begin_t(void);
	
	static nanos6_runtime_info_begin_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_runtime_info_begin_t *) _nanos6_resolve_symbol("nanos6_runtime_info_begin", "runtime info", NULL);
	}
	
	return (*symbol)();
}


void *nanos6_runtime_info_end(void)
{
	typedef void *nanos6_runtime_info_end_t(void);
	
	static nanos6_runtime_info_end_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_runtime_info_end_t *) _nanos6_resolve_symbol("nanos6_runtime_info_end", "runtime info", NULL);
	}
	
	return (*symbol)();
}


void *nanos6_runtime_info_advance(void *runtimeInfoIterator)
{
	typedef void *nanos6_runtime_info_advance_t(void *runtimeInfoIterator);
	
	static nanos6_runtime_info_advance_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_runtime_info_advance_t *) _nanos6_resolve_symbol("nanos6_runtime_info_advance", "runtime info", NULL);
	}
	
	return (*symbol)(runtimeInfoIterator);
}


void nanos6_runtime_info_get(void *runtimeInfoIterator, nanos6_runtime_info_entry_t *entry)
{
	typedef void nanos6_runtime_info_get_t(void *runtimeInfoIterator, nanos6_runtime_info_entry_t *entry);
	
	static nanos6_runtime_info_get_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_runtime_info_get_t *) _nanos6_resolve_symbol("nanos6_runtime_info_get", "runtime info", NULL);
	}
	
	(*symbol)(runtimeInfoIterator, entry);
}


int nanos6_snprint_runtime_info_entry_value(char *str, size_t size, nanos6_runtime_info_entry_t const *entry)
{
	typedef int nanos6_snprint_runtime_info_entry_value_t(char *str, size_t size, nanos6_runtime_info_entry_t const *entry);
	
	static nanos6_snprint_runtime_info_entry_value_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_snprint_runtime_info_entry_value_t *) _nanos6_resolve_symbol("nanos6_snprint_runtime_info_entry_value", "runtime info", NULL);
	}
	
	return (*symbol)(str, size, entry);
}

#pragma GCC visibility pop

