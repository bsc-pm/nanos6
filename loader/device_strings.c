/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2018-2019 Barcelona Supercomputing Center (BSC)
*/

#include <stddef.h>
#include "api/nanos6/task-instantiation.h"

#pragma GCC visibility push(default)

char const * const nanos6_hostcpu_device_name = "HOST";
char const * const nanos6_cuda_device_name = "CUDA";
char const * const nanos6_opencl_device_name = "OPENCL";
char const * const nanos6_cluster_device_name = "CLUSTER";
char const * const nanos6_fpga_device_name = "FPGA";

#pragma GCC visibility pop

