/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#include "cuda-ndrange.hpp"

__global__ void ndrangeKernel(
	int n1, int n2, int n3,
	int t1, int t2, int t3,
	int *gridSizes, int *blockSizes
) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		gridSizes[0] = gridDim.x;
		gridSizes[1] = gridDim.y;
		gridSizes[2] = gridDim.z;
		blockSizes[0] = blockDim.x;
		blockSizes[1] = blockDim.y;
		blockSizes[2] = blockDim.z;
	}
}
