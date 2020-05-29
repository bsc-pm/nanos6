/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "TestAnyProtocolProducer.hpp"

#define N 100

TestAnyProtocolProducer tap;

int main() {
	std::vector<int> v(N);
	int i;

	tap.registerNewTests(1);
	tap.begin();

	#pragma oss task for firstprivate(v)
	for (i = 0; i < N; ++i)
	{}
	#pragma oss taskwait

	tap.evaluate(true, "Program finished correctly");
	tap.end();
	return 0;
}
