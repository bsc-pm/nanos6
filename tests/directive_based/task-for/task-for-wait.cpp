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

	// This test is redundant because the taskfor  already implements the wait
	// clause semantic. But it is useful to check that setting the wait clause
	// does not break anything in the runtime

	#pragma oss task for chunksize(1) concurrent(var) wait
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			++var;
		}
	}

	#pragma oss task in(var)
	{
		tap.evaluate(var.load() == N*N, "Program finished correctly");
	}

	#pragma oss taskwait

	tap.end();
	return 0;
}
