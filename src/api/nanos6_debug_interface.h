#ifndef NANOS6_DEBUG_INTERFACE_H
#define NANOS6_DEBUG_INTERFACE_H


#ifdef __cplusplus
extern "C" {
#endif

char const *nanos_get_runtime_version();
char const *nanos_get_runtime_branch();
char const *nanos_get_runtime_compiler_version();
char const *nanos_get_runtime_compiler_flags();

long nanos_get_num_cpus();

#ifdef __cplusplus
}
#endif

#endif // NANOS6_DEBUG_INTERFACE_H
