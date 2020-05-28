/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "hardware/device/cuda/CUDAAccelerator.hpp"
#include "tasks/Task.hpp"

#include <nanos6/cuda_device.h>

// If called from the context of a CUDA device outlined task function,
// returns the CUDA stream handle the runtime has allocated to the task.
// This is useful to be able to encapsulate e.g. a cuBLAS library call
// inside an OmpSs CUDA task.
extern "C"
cudaStream_t nanos6_get_current_cuda_stream(void)
{
	Task *currentTask = CUDAAccelerator::getCurrentTask();
	nanos6_cuda_device_environment_t &env =	currentTask->getDeviceEnvironment().cuda;
	return env.stream;
}
