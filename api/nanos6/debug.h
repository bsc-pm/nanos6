/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_DEBUG_H
#define NANOS6_DEBUG_H


#include "major.h"


#pragma GCC visibility push(default)

#ifdef __cplusplus
extern "C" {
#endif


//! \brief returns the path of the runtime
//! For instance "/apps/PM/ompss/git/lib/libnanos6-optimized.so"
//! \note This is actually implemented by the loader itself
char const *nanos6_get_runtime_path(void);

//! \brief returns a string that describes the version of the runtime
//! For instance "2015-11-13 11:27:48 +0100 ff1221e"
char const *nanos6_get_runtime_version(void);

//! \brief returns a string that describes the copyright of the runtime
//! For instance "Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)"
char const *nanos6_get_runtime_copyright(void);

//! \brief returns a short string that describes the license of the runtime
char const *nanos6_get_runtime_license(void);

//! \brief returns the full text of the license of the runtime
char const *nanos6_get_runtime_full_license(void);

//! \brief returns a string with the name of the branch of the runtime
//! For instance "master"
char const *nanos6_get_runtime_branch(void);

//! \brief returns a string containing code patches relative to the runtime version
char const *nanos6_get_runtime_patches(void);

//! \brief returns a string describing the version of the compiler used in the runtime
//! For instance "g++ (Debian 5.2.1-23) 5.2.1 20151028"
char const *nanos6_get_runtime_compiler_version(void);

//! \brief returns a string containing the compiler flags used when compiling the runtime
//! For instance "-DNDEBUG   -std=gnu++11 -Wall -Wextra -O3 -flto"
char const *nanos6_get_runtime_compiler_flags(void);

//! \brief wait until the runtime has been fully initialized
//! This function can be called from within used code, most probably from "main"
void nanos6_wait_for_full_initialization(void);

//! \brief get the number of CPUs that were enabled when the program started
unsigned int nanos6_get_num_cpus(void);

//! \brief get the operating system assigned identifier of the CPU where the call to this function originated
long nanos6_get_current_system_cpu(void);

//! \brief get a CPU identifier assigned to the CPU where the call to this function originated that starts from 0 up to nanos6_get_num_cpus(void)-1
unsigned int nanos6_get_current_virtual_cpu(void);

//! \brief enable a previously stopped CPU
void nanos6_enable_cpu(long systemCPUId);

//! \brief disable an enabled CPU
void nanos6_disable_cpu(long systemCPUId);

typedef enum {
	nanos6_invalid_cpu_status,
	nanos6_starting_cpu,
	nanos6_enabling_cpu,
	nanos6_enabled_cpu,
	nanos6_disabling_cpu,
	nanos6_disabled_cpu
} nanos6_cpu_status_t;

//! \brief retrieve the runtime view of a given CPU identified by the identifier given by the operating system
nanos6_cpu_status_t nanos6_get_cpu_status(long systemCPUId);

// void nanos6_wait_until_task_starts(void *taskHandle);
// long nanos6_get_system_cpu_of_task(void *taskHandle);

//! \brief obtain an iterator to the beginning of the list of CPUs handled by the runtime
void *nanos6_cpus_begin(void);

//! \brief obtain an iterator past the end of the list of CPUs handled by the runtime
void *nanos6_cpus_end(void);

//! \brief advance an iterator of the list of CPUs to the following element
void *nanos6_cpus_advance(void *cpuIterator);

//! \brief retrieve the operating system assigned identifier to the CPU pointed to by the iterator
long nanos6_cpus_get(void *cpuIterator);

//! \brief retrieve the runtime-assigned identifier to the CPU pointed to by the iterator
long nanos6_cpus_get_virtual(void *cpuIterator);


#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif /* NANOS6_DEBUG_H */
