/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2025 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_GRID_HPP
#define CUDA_GRID_HPP

#ifdef __cplusplus
extern "C"
{
#endif

#pragma oss task device(cuda) grid(3, g1, g2, g3, t1, t2, t3) out([3]gridSizes) out([3]blockSizes)
__global__ void gridKernel(
	int g1, int g2, int g3,
	int t1, int t2, int t3,
	int *gridSizes, int *blockSizes);

#ifdef __cplusplus
}
#endif

#endif // CUDA_GRID_HPP
