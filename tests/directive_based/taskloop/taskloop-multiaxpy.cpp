/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include <algorithm>
#include <cstdlib>

#include "TestAnyProtocolProducer.hpp"

#define TOTALSIZE (128*1024*1024)
#define BLOCKSIZE (1024*1024)
#define GRAINSIZE (1024*1024)
#define ITERATIONS (10)

TestAnyProtocolProducer tap;

static void initialize(double *data, double value, long N, long BS) {
	for (long i = 0; i < N; i += BS) {
		long elements = std::min(BS, N - i);

		#pragma oss task out(data[i;elements])
		for (long j = 0; j < elements; ++j) {
			data[i + j] = value;
		}
	}
}

static void axpy(const double *x, double *y, double alpha, long N, long BS, long GS) {
	#pragma oss taskloop grainsize(GS)
	for (long i = 0; i < N; i++) {
		y[i] += alpha * x[i];
	}
}

static bool validate(double *y, long N, long BS, double expectedValue) {
	int errors = 0;

	for (long i = 0; i < N; i += BS) {
		long elements = std::min(BS, N - i);

		#pragma oss task in(y[i;elements]) reduction(+:errors)
		for (long j = 0; j < elements; ++j) {
			if (y[i + j] != expectedValue) {
				errors += 1;
				break;
			}
		}
	}
	#pragma oss taskwait

	return (errors == 0);
}

int main() {
	long n = TOTALSIZE;
	long bs = BLOCKSIZE;
	long gs = GRAINSIZE;
	long its = ITERATIONS;

	// Initialization
	double *x = new double[n];
	double *y = new double[n];

	tap.registerNewTests(1);
	tap.begin();

	initialize(x, 1.0, n, bs);
	initialize(y, 0.0, n, bs);
	#pragma oss taskwait

	// Main algorithm
	for (int iteration = 0; iteration < its; iteration++) {
		axpy(x, y, 1.0, n, bs, gs);
		#pragma oss taskwait
	}

	// Validation
	bool validates = validate(y, n, bs, its);

	tap.evaluate(validates, "The result of the multiaxpy program is correct");
	tap.end();

	delete[] x;
	delete[] y;
	return 0;
}
