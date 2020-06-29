/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_SAXPY_HPP
#define CUDA_SAXPY_HPP

#ifdef __cplusplus
extern "C"
{
#endif

#pragma oss task in([n]x) inout([n]y) if(!if0) device(cuda) ndrange(1, n, 128)
__global__ void saxpyCUDAKernel(long int n, double a, const double* x, double* y, bool if0);

#ifdef __cplusplus
}
#endif

#endif // CUDA_SAXPY_HPP
