/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_CUDA_DEVICE_H
#define NANOS6_CUDA_DEVICE_H

#include "major.h"


#pragma GCC visibility push(default)


// NOTE: The full version depends also on nanos6_major_api
//       That is:   nanos6_major_api . nanos6_cuda_device_api
enum nanos6_cuda_device_api_t { nanos6_cuda_device_api = 1 };

#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef struct {
	cudaStream_t stream;
} nanos6_cuda_device_environment_t;


#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop

#endif /* NANOS6_CUDA_DEVICE_H */

