/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include <stdio.h>

#include "cuda-saxpy.hpp"


__global__ void saxpyCUDAKernel(long int n, double a, const double* x, double* y, bool if0)
{
	long int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) y[i] = a * x[i] + y[i];
}
