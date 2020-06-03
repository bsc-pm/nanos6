/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef DEVICE_ENVIRONMENT_HPP
#define DEVICE_ENVIRONMENT_HPP

// Use this header to define the device environment -used by Mercurium- for various devices
// implementations.

#include <config.h>

#if USE_CUDA
#include <nanos6/cuda_device.h>
#endif

#if USE_OPENACC
#include <nanos6/openacc_device.h>
#endif

//! Contains the device environment based on configured device types to determine max size
union DeviceEnvironment {
#if USE_CUDA
		nanos6_cuda_device_environment_t cuda;
#endif
#if USE_OPENACC
		nanos6_openacc_device_environment_t openacc;
#endif
};

#endif // DEVICE_ENVIRONMENT_HPP

