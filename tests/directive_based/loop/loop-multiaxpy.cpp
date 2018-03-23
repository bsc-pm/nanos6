/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <algorithm>
#include <cstdlib>

#include "TestAnyProtocolProducer.hpp"

#define TOTALSIZE (128*1024*1024)
#define CHUNKSIZE (2048)
#define ITERATIONS (10)

TestAnyProtocolProducer tap;

#pragma oss task label(init block)
static void initialize_task(double *data, long size, double value) {
	for (long i=0; i<size; i++) {
		data[i] = value;
	}
}


static void initialize(double *data, double value, long N, long TS) {
	for (long i=0; i < N; i+=TS) {
		long elements = std::min(TS, N-i);
		initialize_task(&data[i], elements, value);
	}
}


static void axpy(double *x, double *y, double alpha, long N, long TS) {
	#pragma oss loop chunksize(TS)
	for (long i=0; i < N; ++i) {
		y[i] += alpha * x[i];
	}
}


static bool validate(double *y, long N, double expectedValue) {
	for (long i=0; i < N; ++i) {
		if (y[i] != expectedValue) {
			return false;
		}
	}
	return true;
}


bool validScheduler() {
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
	long ts = CHUNKSIZE;
	long its = ITERATIONS;
	
	// Initialization
	double *x = new double[n];
	double *y = new double[n];
	initialize(x, 1.0, n, ts);
	initialize(y, 0.0, n, ts);
	#pragma oss taskwait
	
	tap.registerNewTests(1);
	tap.begin();
	
	// Main algorithm
	for (int iteration=0; iteration < its; iteration++) {
		axpy(x, y, 1.0, n, ts);
		#pragma oss taskwait
	}
	
	// Validation
	double expectedValue = its;
	bool validates = validate(y, n, expectedValue);
	
	tap.evaluate(validates, "The result of the multiaxpy program is correct");
	
	tap.end();
	
	return 0;
}
