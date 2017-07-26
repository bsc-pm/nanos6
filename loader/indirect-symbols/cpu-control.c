/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"
#include "api/nanos6/debug.h"


long nanos_get_current_system_cpu()
{
	typedef long nanos_get_current_system_cpu_t();
	
	static nanos_get_current_system_cpu_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_get_current_system_cpu_t *) _nanos6_resolve_symbol("nanos_get_current_system_cpu", "cpu control", NULL);
	}
	
	return (*symbol)();
}


unsigned int nanos_get_current_virtual_cpu()
{
	typedef unsigned int nanos_get_current_virtual_cpu_t();
	
	static nanos_get_current_virtual_cpu_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_get_current_virtual_cpu_t *) _nanos6_resolve_symbol("nanos_get_current_virtual_cpu", "cpu control", NULL);
	}
	
	return (*symbol)();
}


void nanos_enable_cpu(long systemCPUId)
{
	typedef void nanos_enable_cpu_t(long systemCPUId);
	
	static nanos_enable_cpu_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_enable_cpu_t *) _nanos6_resolve_symbol("nanos_enable_cpu", "cpu control", NULL);
	}
	
	(*symbol)(systemCPUId);
}


void nanos_disable_cpu(long systemCPUId)
{
	typedef void nanos_disable_cpu_t(long systemCPUId);
	
	static nanos_disable_cpu_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_disable_cpu_t *) _nanos6_resolve_symbol("nanos_disable_cpu", "cpu control", NULL);
	}
	
	(*symbol)(systemCPUId);
}


nanos_cpu_status_t nanos_get_cpu_status(long systemCPUId)
{
	typedef nanos_cpu_status_t nanos_get_cpu_status_t(long systemCPUId);
	
	static nanos_get_cpu_status_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_get_cpu_status_t *) _nanos6_resolve_symbol("nanos_get_cpu_status", "cpu control", NULL);
	}
	
	return (*symbol)(systemCPUId);
}


void *nanos_cpus_begin(void)
{
	typedef void *nanos_cpus_begin_t(void);
	
	static nanos_cpus_begin_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_cpus_begin_t *) _nanos6_resolve_symbol("nanos_cpus_begin", "cpu control", NULL);
	}
	
	return (*symbol)();
}


void *nanos_cpus_end(void)
{
	typedef void *nanos_cpus_end_t(void);
	
	static nanos_cpus_end_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_cpus_end_t *) _nanos6_resolve_symbol("nanos_cpus_end", "cpu control", NULL);
	}
	
	return (*symbol)();
}


void *nanos_cpus_advance(void *cpuIterator)
{
	typedef void *nanos_cpus_advance_t(void *cpuIterator);
	
	static nanos_cpus_advance_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_cpus_advance_t *) _nanos6_resolve_symbol("nanos_cpus_advance", "cpu control", NULL);
	}
	
	return (*symbol)(cpuIterator);
}


long nanos_cpus_get(void *cpuIterator)
{
	typedef long nanos_cpus_get_t(void *cpuIterator);
	
	static nanos_cpus_get_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_cpus_get_t *) _nanos6_resolve_symbol("nanos_cpus_get", "cpu control", NULL);
	}
	
	return (*symbol)(cpuIterator);
}


long nanos_cpus_get_virtual(void *cpuIterator)
{
	typedef long nanos_cpus_get_virtual_t(void *cpuIterator);
	
	static nanos_cpus_get_virtual_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_cpus_get_virtual_t *) _nanos6_resolve_symbol("nanos_cpus_get_virtual", "cpu control", NULL);
	}
	
	return (*symbol)(cpuIterator);
}

