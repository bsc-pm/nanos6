/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#include "cuda-shmem.hpp"

__global__ void shmemKernel(int n, int bs)
{
	extern __shared__ int shmem[];

	int *arrayA = (int *) &shmem[0];
	double *arrayB = (double *) &arrayA[bs];

	int index = threadIdx.x;
	arrayA[index] = index;
	arrayB[index] = index;
}
