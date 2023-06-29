/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#include "cuda-kernargs.hpp"

__global__ void kernargsKernel(int n, int bs, const int *ptr,
	int param1, float param2, int32_t param3, double param4,
	uint64_t param5, int8_t param6, size_t param7, void *param8,
	int *sentinel, int *gError
) {
	// Check if there is any mismatch
	int error = (param1 != paramVal1 || param2 != paramVal2
		|| param3 != paramVal3 || param4 != paramVal4
		|| param5 != paramVal5 || param6 != paramVal6
		|| param7 != paramVal7 || param8 != (void *) ptr);

	// Add the error boolean to the global error counter
	atomicAdd(gError, error);
}
