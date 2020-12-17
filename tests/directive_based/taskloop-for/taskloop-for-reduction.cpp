/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "TestAnyProtocolProducer.hpp"

#define N 10000
#define N_ARR 100

TestAnyProtocolProducer tap;

int testArr[N_ARR] = { 0 };

int main() {
	int i;
	int test = 0;

	tap.registerNewTests(2);
	tap.begin();

	#pragma oss taskloop for reduction(+: test)
	for (i = 0; i < N; ++i) {
		test++;
	}

	#pragma oss taskloop for reduction(+: [100]testArr)
	for (i = 0; i < N; ++i) {
		for (int j = 0; j < N_ARR; ++j)
			testArr[j] += j;
	}

	#pragma oss taskwait

	bool correct = true;
	if (test != N)
		correct = false;

	tap.evaluate(correct, "Single element reduction is correct");
	correct = true;

	for (int j = 0; j < N_ARR; ++j) {
		if (testArr[j] != j * N) {
			correct = false;
			break;
		}
	}

	tap.evaluate(correct, "Array reduction is correct");

	tap.end();
	return 0;
}
