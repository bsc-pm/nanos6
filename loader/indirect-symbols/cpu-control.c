/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"
#include "api/nanos6/debug.h"


#pragma GCC visibility push(default)

long nanos6_get_current_system_cpu(void)
{
	typedef long nanos6_get_current_system_cpu_t(void);
	
	static nanos6_get_current_system_cpu_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_get_current_system_cpu_t *) _nanos6_resolve_symbol("nanos6_get_current_system_cpu", "cpu control", NULL);
	}
	
	return (*symbol)();
}


unsigned int nanos6_get_current_virtual_cpu(void)
{
	typedef unsigned int nanos6_get_current_virtual_cpu_t(void);
	
	static nanos6_get_current_virtual_cpu_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_get_current_virtual_cpu_t *) _nanos6_resolve_symbol("nanos6_get_current_virtual_cpu", "cpu control", NULL);
	}
	
	return (*symbol)();
}


void nanos6_enable_cpu(long systemCPUId)
{
	typedef void nanos6_enable_cpu_t(long systemCPUId);
	
	static nanos6_enable_cpu_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_enable_cpu_t *) _nanos6_resolve_symbol("nanos6_enable_cpu", "cpu control", NULL);
	}
	
	(*symbol)(systemCPUId);
}


void nanos6_disable_cpu(long systemCPUId)
{
	typedef void nanos6_disable_cpu_t(long systemCPUId);
	
	static nanos6_disable_cpu_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_disable_cpu_t *) _nanos6_resolve_symbol("nanos6_disable_cpu", "cpu control", NULL);
	}
	
	(*symbol)(systemCPUId);
}


nanos6_cpu_status_t nanos6_get_cpu_status(long systemCPUId)
{
	typedef nanos6_cpu_status_t nanos6_get_cpu_status_t(long systemCPUId);
	
	static nanos6_get_cpu_status_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_get_cpu_status_t *) _nanos6_resolve_symbol("nanos6_get_cpu_status", "cpu control", NULL);
	}
	
	return (*symbol)(systemCPUId);
}


void *nanos6_cpus_begin(void)
{
	typedef void *nanos6_cpus_begin_t(void);
	
	static nanos6_cpus_begin_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_cpus_begin_t *) _nanos6_resolve_symbol("nanos6_cpus_begin", "cpu control", NULL);
	}
	
	return (*symbol)();
}


void *nanos6_cpus_end(void)
{
	typedef void *nanos6_cpus_end_t(void);
	
	static nanos6_cpus_end_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_cpus_end_t *) _nanos6_resolve_symbol("nanos6_cpus_end", "cpu control", NULL);
	}
	
	return (*symbol)();
}


void *nanos6_cpus_advance(void *cpuIterator)
{
	typedef void *nanos6_cpus_advance_t(void *cpuIterator);
	
	static nanos6_cpus_advance_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_cpus_advance_t *) _nanos6_resolve_symbol("nanos6_cpus_advance", "cpu control", NULL);
	}
	
	return (*symbol)(cpuIterator);
}


long nanos6_cpus_get(void *cpuIterator)
{
	typedef long nanos6_cpus_get_t(void *cpuIterator);
	
	static nanos6_cpus_get_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_cpus_get_t *) _nanos6_resolve_symbol("nanos6_cpus_get", "cpu control", NULL);
	}
	
	return (*symbol)(cpuIterator);
}


long nanos6_cpus_get_virtual(void *cpuIterator)
{
	typedef long nanos6_cpus_get_virtual_t(void *cpuIterator);
	
	static nanos6_cpus_get_virtual_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_cpus_get_virtual_t *) _nanos6_resolve_symbol("nanos6_cpus_get_virtual", "cpu control", NULL);
	}
	
	return (*symbol)(cpuIterator);
}


#pragma GCC visibility pop
