/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_SHMEM_HPP
#define CUDA_SHMEM_HPP

#include <cstdint>
#include <limits>

#pragma oss task device(cuda) ndrange(1, n, bs) shmem(bs*(sizeof(int)+sizeof(double)))
__global__ void shmemKernel(int n, int bs);

#endif // CUDA_SHMEM_HPP
