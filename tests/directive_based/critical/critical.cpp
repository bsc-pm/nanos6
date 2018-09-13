/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/debug.h>

#include <Atomic.hpp>
#include "TestAnyProtocolProducer.hpp"
#include "Timer.hpp"

#include <cassert>


TestAnyProtocolProducer tap;


#define ARRAY_SIZE (1024L * 1024L * 10L)

static long *data;
static long numCPUs;


#pragma oss task label(initialize)
static void initialize(long participant)
{
	long chunkSize = ARRAY_SIZE / numCPUs;
	long start = participant * chunkSize;
	long end = (participant + 1) * chunkSize;
	if (participant == numCPUs-1) {
		end = ARRAY_SIZE;
	}
	
	for (long i=start; i < end; i++) {
		data[i] = 1;
	}
}


static Atomic<int> concurrent_tasks;
static void *critical_handle = 0;
volatile long sum_result = 0;


#pragma oss task label(sum)
static void sum(long participant)
{
	long chunkSize = ARRAY_SIZE / numCPUs;
	long start = participant * chunkSize;
	long end = (participant + 1L) * chunkSize;
	if (participant == numCPUs-1) {
		end = ARRAY_SIZE;
	}
	
	#pragma oss critical
	{
		concurrent_tasks++;
		tap.evaluate(concurrent_tasks == 1, "Check that only one task is in the critical region after entering it");
		for (long i=start; i < end; i++) {
			sum_result += data[i];
		}
		tap.evaluate(concurrent_tasks == 1, "Check that only one task is in the critical region before exiting it");
		concurrent_tasks--;
	}
}


int main(int argc, char **argv) {
	nanos6_wait_for_full_initialization();
	
	numCPUs = nanos6_get_num_cpus();
	
	tap.registerNewTests(numCPUs * 2L + 1);
	
	// Initialize in parallel
	data = new long[ARRAY_SIZE];
	for (int i=0; i < numCPUs ; i++) {
		initialize(i);
	}
	#pragma oss taskwait
	
	tap.begin();
	
	// Sum in tasks but with a critical inside
	concurrent_tasks = 0;
	for (int i=0; i < numCPUs ; i++) {
		sum(i);
	}
	#pragma oss taskwait
	
	tap.evaluate(sum_result == ARRAY_SIZE, "Check that the result is correct");
	tap.emitDiagnostic<>("Expected result: ", ARRAY_SIZE);
	tap.emitDiagnostic<>("Actual result: ", sum_result);
	tap.end();
	
	return 0;
}

