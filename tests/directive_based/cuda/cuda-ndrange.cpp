/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#include "cuda-ndrange.hpp"

#include "TestAnyProtocolProducer.hpp"
#include "UtilsCUDA.hpp"


TestAnyProtocolProducer tap;

int main()
{
	tap.registerNewTests(1);
	tap.begin();

	const int total[3] = { 1026, 64, 16 };
	const int threads[3] = { 32, 8, 2 };
	int blocks[3];

	for (int d = 0; d < 3; d++)
		blocks[d] = (total[d] % threads[d] == 0)
			? (total[d] / threads[d]) : (total[d] / threads[d] + 1);

	// Allocate CUDA unified memory
	int *outGridSizes, *outBlockSizes;
	CUDA_CHECK(cudaMallocManaged(&outGridSizes, 3 * sizeof(int), cudaMemAttachGlobal));
	CUDA_CHECK(cudaMallocManaged(&outBlockSizes, 3 * sizeof(int), cudaMemAttachGlobal));

	std::memset(outGridSizes, 0, 3 * sizeof(int));
	std::memset(outBlockSizes, 0, 3 * sizeof(int));

	// Launch a single kernel with specific ndrange values
	ndrangeKernel(total[0], total[1], total[2], threads[0], threads[1], threads[2],
		outGridSizes, outBlockSizes);
	#pragma oss taskwait

	// Check that the launched kernel executed the expected blocks and threads
	bool validates = true;
	for (int d = 0; d < 3; d++) {
		if (outGridSizes[d] != blocks[d])
			validates = false;
		if (outBlockSizes[d] != threads[d])
			validates = false;
	}

	tap.evaluate(validates, "The ndrange parameters are respected");
	tap.end();

	CUDA_CHECK(cudaFree(outGridSizes));
	CUDA_CHECK(cudaFree(outBlockSizes));

	return 0;
}
