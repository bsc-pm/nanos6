/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_DEVICE_STRINGS_C
#define NANOS6_DEVICE_STRINGS_C


#include <stddef.h>
#include "api/nanos6/task-instantiation.h" 

#pragma GCC visibility push(default)

#ifdef __cplusplus
extern "C" {
#endif

char const * const nanos6_hostcpu_device_name = "HOST";
char const * const nanos6_cuda_device_name = "CUDA";
char const * const nanos6_opencl_device_name = "OPENCL";

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif /* NANOS6_DEVICE_STRINGS_C */
