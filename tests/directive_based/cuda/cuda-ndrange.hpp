/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_NDRANGE_HPP
#define CUDA_NDRANGE_HPP

#ifdef __cplusplus
extern "C"
{
#endif

#pragma oss task device(cuda) ndrange(3, n1, n2, n3, t1, t2, t3) out([3]gridSizes) out([3]blockSizes)
__global__ void ndrangeKernel(
	int n1, int n2, int n3,
	int t1, int t2, int t3,
	int *gridSizes, int *blockSizes);

#ifdef __cplusplus
}
#endif

#endif // CUDA_NDRANGE_HPP
