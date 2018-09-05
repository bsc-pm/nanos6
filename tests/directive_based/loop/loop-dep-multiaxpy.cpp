/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <algorithm>
#include <cstdlib>

#include "TestAnyProtocolProducer.hpp"

#define TOTALSIZE (128*1024*1024)
#define BLOCKSIZE (1024*1024)
#define CHUNKSIZE (256)
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

static void axpy(const double *x, double *y, double alpha, long N, long BS, long CS) {
	for (long i = 0; i < N; i += BS) {
		long elements = std::min(BS, N - i);
		
		#pragma oss loop in(x[i;elements]) inout(y[i;elements]) chunksize(CS)
		for (long j = 0; j < elements; ++j) {
			y[i + j] += alpha * x[i + j];
		}
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

static bool validScheduler() {
	// Taskloop is only supported by Naive and FIFO schedulers
	char const *schedulerName = getenv("NANOS6_SCHEDULER");
	if (schedulerName != 0) {
		std::string scheduler(schedulerName);
		if (scheduler == "naive" || scheduler == "fifo") {
			return true;
		}
	}
	return false;
}

int main() {
	if (!validScheduler()) {
		tap.registerNewTests(1);
		tap.begin();
		tap.skip("This test does not work with this scheduler");
		tap.end();
		return 0;
	}
	
	long n = TOTALSIZE;
	long bs = BLOCKSIZE;
	long cs = CHUNKSIZE;
	long its = ITERATIONS;
	
	// Initialization
	double *x = new double[n];
	double *y = new double[n];
	
	tap.registerNewTests(1);
	tap.begin();
	
	initialize(x, 1.0, n, bs);
	initialize(y, 0.0, n, bs);
	
	// Main algorithm
	for (int iteration = 0; iteration < its; iteration++) {
		axpy(x, y, 1.0, n, bs, cs);
	}
	#pragma oss taskwait
	
	// Validation
	bool validates = validate(y, n, bs, its);
	
	tap.evaluate(validates, "The result of the multiaxpy program is correct");
	tap.end();
	
	delete[] x;
	delete[] y;
	return 0;
}
