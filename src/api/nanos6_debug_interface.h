#ifndef NANOS6_DEBUG_INTERFACE_H
#define NANOS6_DEBUG_INTERFACE_H


#ifdef __cplusplus
extern "C" {
#endif

char const *nanos_get_runtime_version(void);
char const *nanos_get_runtime_branch(void);
char const *nanos_get_runtime_compiler_version(void);
char const *nanos_get_runtime_compiler_flags(void);

void nanos_wait_for_full_initialization(void);

long nanos_get_num_cpus(void);

long nanos_get_current_system_cpu();
void nanos_enable_cpu(long systemCPUId);
void nanos_disable_cpu(long systemCPUId);

typedef enum {
	nanos_invalid_cpu_status,
	nanos_starting_cpu,
	nanos_enabling_cpu,
	nanos_enabled_cpu,
	nanos_disabling_cpu,
	nanos_disabled_cpu
} nanos_cpu_status_t;

nanos_cpu_status_t nanos_get_cpu_status(long systemCPUId);

// void nanos_wait_until_task_starts(void *taskHandle);
// long nanos_get_system_cpu_of_task(void *taskHandle);

void *nanos_cpus_begin(void);
void *nanos_cpus_end(void);
void *nanos_cpus_advance(void *cpuIterator);
long nanos_cpus_get(void *cpuIterator);


#ifdef __cplusplus
}
#endif

#endif // NANOS6_DEBUG_INTERFACE_H
