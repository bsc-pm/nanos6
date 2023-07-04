/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_KERNARGS_HPP
#define CUDA_KERNARGS_HPP

#include <cstdint>
#include <limits>

constexpr int      paramVal1 = 1;
constexpr float    paramVal2 = 2.0f;
constexpr int32_t  paramVal3 = 3;
constexpr double   paramVal4 = std::numeric_limits<double>::min();
constexpr uint64_t paramVal5 = std::numeric_limits<uint64_t>::max();
constexpr int8_t   paramVal6 = 6;
constexpr size_t   paramVal7 = std::numeric_limits<size_t>::max();

// NOTE: This kernel also tests that the prefecthing of non-CUDA memory does not
// make the runtime to abort the execution. The sentinel variable is actually a
// global variable and may not be prefetchable to the device. By default, the
// runtime should attempt to prefetch it to the device and ignore the error

#pragma oss task inout(*sentinel) in(ptr[0;n]) device(cuda) ndrange(1, n, bs)
__global__ void kernargsKernel(int n, int bs, const int *ptr,
	int param1, float param2, int32_t param3, double param4,
	uint64_t param5, int8_t param6, size_t param7, void *param8,
	int *sentinel, int *error);

#endif // CUDA_KERNARGS_HPP
