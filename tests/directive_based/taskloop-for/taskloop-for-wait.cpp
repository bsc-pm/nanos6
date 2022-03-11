/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
*/

#include "Atomic.hpp"
#include "TestAnyProtocolProducer.hpp"

#define N 1000

TestAnyProtocolProducer tap;

int main() {
	Atomic<int> var(0);

	tap.registerNewTests(1);
	tap.begin();

	#pragma oss taskloop for grainsize(10) chunksize(1) concurrent(var) wait
	for (int i = 0; i < N*N; ++i) {
		++var;
	}

	#pragma oss task in(var)
	{
		tap.evaluate(var.load() == N*N, "Program finished correctly");
	}

	#pragma oss taskwait

	tap.end();
	return 0;
}
