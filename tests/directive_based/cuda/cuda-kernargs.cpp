/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#include <cstdlib>

#include <cuda_runtime.h>

#include "cuda-kernargs.hpp"

#include "TestAnyProtocolProducer.hpp"
#include "UtilsCUDA.hpp"


TestAnyProtocolProducer tap;
int sentinel;

int main()
{
	tap.registerNewTests(1);
	tap.begin();

	const int n = 1024;
	const int bs = 128;

	// Allocate CUDA unified memory
	int *outError = nullptr;
	int *ptr = nullptr;
	CUDA_CHECK(cudaMallocManaged(&ptr, n * sizeof(int), cudaMemAttachGlobal));
	CUDA_CHECK(cudaMallocManaged(&outError, sizeof(int), cudaMemAttachGlobal));

	// Initialize the error to zero
	*outError = 0;

	// Launch a single kernel with the specific parameters
	kernargsKernel(n, bs, ptr, paramVal1, paramVal2,
		paramVal3, paramVal4, paramVal5, paramVal6,
		paramVal7, ptr, &sentinel, outError);
	#pragma oss taskwait

	tap.evaluate((*outError == 0), "The kernel parameters are respected");
	tap.end();

	CUDA_CHECK(cudaFree(ptr));
	CUDA_CHECK(cudaFree(outError));

	return 0;
}
