/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <algorithm>
#include <cstdlib>

#ifndef __ICC
#include <atomic>
#endif

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


#pragma oss task label(block validation)
static void validate_task(double *y, long size, double expectedValue, volatile bool *validates) {
	for (long i=0; i<size; i++) {
		if (y[i] != expectedValue) {
			*validates = false;
#ifndef __ICC
			std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
			return;
		}
		if (!*validates) {
			return;
		}
	}
}


static void validate(double *y, long N, long TS, double expectedValue, volatile bool *validates) {
	for (long i=0; i < N; i+=TS) {
		long elements = std::min(TS, N-i);
		validate_task(&y[i], elements, expectedValue, validates);
	}
}


bool validScheduler() {
	// Taskloop is only supported by Naive and FIFO schedulers
	char const *schedulerName = getenv("NANOS6_SCHEDULER");
	if (schedulerName != nullptr) {
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
	volatile bool validates = true;
	double expectedValue = its;
	validate(y, n, ts, expectedValue, &validates);
	#pragma oss taskwait
	
	tap.evaluate(validates, "The result of the multiaxpy program is correct");
	
	tap.end();
	
	return 0;
}
