/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#include <cstdlib>

#include <cuda_runtime.h>

#include "cuda-shmem.hpp"

#include "TestAnyProtocolProducer.hpp"


TestAnyProtocolProducer tap;

int main() {
	tap.registerNewTests(1);
	tap.begin();

	const int n = 1*1024*1024;
	const int bs = 128;

	// Launch a single kernel with the specific shared memory
	shmemKernel(n, bs);
	#pragma oss taskwait

	tap.success("The shared memory clause is respected");
	tap.end();

	return 0;
}
