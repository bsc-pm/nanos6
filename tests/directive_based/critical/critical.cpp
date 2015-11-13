#include <nanos6_debug_interface.h>

#include "infrastructure/ProgramLifecycle.hpp"
#include "infrastructure/TestAnyProtocolProducer.hpp"
#include "infrastructure/Timer.hpp"

#include <atomic>
#include <cassert>


extern TestAnyProtocolProducer tap;


void shutdownTests()
{
}


#define ARRAY_SIZE (1024L * 1024L * 10L)

static long *data;
static long numCPUs;


#pragma oss task
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


static std::atomic<int> concurrent_tasks;
static void *critical_handle = nullptr;
volatile long sum_result = 0;


#pragma oss task
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
	nanos_wait_for_full_initialization();
	
	numCPUs = nanos_get_num_cpus();
	
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
	
	return 0;
}

