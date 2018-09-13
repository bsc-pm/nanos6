/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/debug.h>

void nanos6_wait_for_full_initialization(void)
{
}

unsigned int nanos6_get_num_cpus()
{
	return 1;
}

long nanos6_get_current_system_cpu(void)
{
	return 0;
}

unsigned int nanos6_get_current_virtual_cpu(void)
{
	return 0;
}

void nanos6_enable_cpu(__attribute__((unused)) long systemCPUId)
{
}

void nanos6_disable_cpu(__attribute__((unused)) long systemCPUId)
{
}


nanos6_cpu_status_t nanos6_get_cpu_status(__attribute__((unused)) long systemCPUId)
{
	return nanos6_enabled_cpu;
}


void *nanos6_cpus_begin(void)
{
	return 0;
}

void *nanos6_cpus_end(void)
{
	return 0;
}

void *nanos6_cpus_advance(__attribute__((unused)) void *cpuIterator)
{
	return 0;
}

long nanos6_cpus_get(__attribute__((unused)) void *cpuIterator)
{
	return 0;
}

