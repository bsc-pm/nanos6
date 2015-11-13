#include "api/nanos6_debug_interface.h"

void nanos_wait_for_full_initialization(void)
{
}

long nanos_get_num_cpus()
{
	return 1;
}

long nanos_supports_cpu_management(void)
{
	return 0;
}

long nanos_get_current_system_cpu(void)
{
	return 0;
}

void nanos_enable_cpu(__attribute__((unused)) long systemCPUId)
{
}

void nanos_disable_cpu(__attribute__((unused)) long systemCPUId)
{
}


nanos_cpu_status_t nanos_get_cpu_status(__attribute__((unused)) long systemCPUId)
{
	return nanos_enabled_cpu;
}


void *nanos_cpus_begin(void)
{
	return 0;
}

void *nanos_cpus_end(void)
{
	return 0;
}

void *nanos_cpus_advance(__attribute__((unused)) void *cpuIterator)
{
	return 0;
}

long nanos_cpus_get(__attribute__((unused)) void *cpuIterator)
{
	return 0;
}

