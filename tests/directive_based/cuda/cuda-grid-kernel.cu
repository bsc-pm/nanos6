/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2025 Barcelona Supercomputing Center (BSC)
*/

#include "cuda-grid.hpp"

__global__ void gridKernel(
	int g1, int g2, int g3,
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
