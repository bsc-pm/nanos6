/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2023 Barcelona Supercomputing Center (BSC)
*/

#include <algorithm>
#include <cstdlib>

#include <cuda_runtime.h>

#include "cuda-saxpy.hpp"

#include "TestAnyProtocolProducer.hpp"
#include "UtilsCUDA.hpp"


#define TOTALSIZE  (4*1024*1024)
#define BLOCKSIZE  (4096)
#define ITERATIONS (50)


TestAnyProtocolProducer tap;

#pragma oss task out([BS]x) out([BS]y)
void initializeChunk(long int BS, long int start, double *x, double *y) {
	for (long int i = 0; i < BS; ++i) {
		x[i] = start + i;
		y[i] = start + i + 2;
	}
}

void initialize(long int N, long int BS, double *x, double *y) {
	for (long int i = 0; i < N; i += BS) {
		initializeChunk(BS, i, &x[i], &y[i]);
	}
}

void saxpy(long int N, long int BS, double a, double *x, double *y, bool if0) {
	for (long int i = 0; i < N; i += BS) {
		saxpyCUDAKernel(BS, a, &x[i], &y[i], if0);
	}
}

bool validate(long int N, double a, long int ITS, double *y) {
	for (long int i = 0; i < N; ++i) {
		if (y[i] != a * i * ITS + (i + 2)) {
			// There may be doubleing point precision errors in large numbers
			return false;
		}
	}
	return true;
}

int main()
{
	tap.registerNewTests(1);
	tap.begin();

	// Saxpy parameters
	const double a = 5;
	const int N = TOTALSIZE;
	const int BS = BLOCKSIZE;
	const int ITS = ITERATIONS;

	// Allocate CUDA unified memory
	double *x, *y;
	CUDA_CHECK(cudaMallocManaged(&x, N * sizeof(double), cudaMemAttachGlobal));
	CUDA_CHECK(cudaMallocManaged(&y, N * sizeof(double), cudaMemAttachGlobal));

	initialize(N, BS, x, y);

	for (int i = 0; i < ITS; ++i) {
		bool if0 = (i == ITS / 2);
		saxpy(N, BS, a, x, y, if0);
	}
	#pragma oss taskwait

	bool validates = validate(N, a, ITS, y);

	tap.evaluate(validates, "The result of the multiaxpy program is correct");
	tap.end();

	CUDA_CHECK(cudaFree(x));
	CUDA_CHECK(cudaFree(y));

	return 0;
}
