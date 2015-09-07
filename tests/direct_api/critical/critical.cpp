#include "api/nanos6_rt_interface.h"

#include "tests/infrastructure/ProgramLifecycle.hpp"
#include "tests/infrastructure/TestAnyProtocolProducer.hpp"
#include "tests/infrastructure/Timer.hpp"

#include "executors/threads/ThreadManagerDebuggingInterface.hpp"

#include <atomic>
#include <cassert>

#include <sched.h>


#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define __FILE_LINE__ (__FILE__ ":" TOSTRING(__LINE__))


extern TestAnyProtocolProducer tap;


void shutdownTests()
{
}


#define ARRAY_SIZE (1024L * 1024L * 10L)

static long *data;
static long numCPUs;


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

static void initialize_wrapper(void *argsBlock)
{
	long *participantPointer = (long *) argsBlock;
	
	initialize(*participantPointer);
}

static void initialize_register_depinfo(__attribute__((unused)) void *handler, __attribute__((unused)) void *argsBlock)
{
}

static void initialize_register_copies(__attribute__((unused)) void *handler, __attribute__((unused)) void *argsBlock)
{
}

static nanos_task_info initialize_info = {
	initialize_wrapper,
	initialize_register_depinfo,
	initialize_register_copies,
	"initialize",
	"initialize_source_line"
};


static std::atomic<int> concurrent_tasks;
static void *critical_handle = nullptr;
volatile long sum_result = 0;

static void sum(long participant)
{
	long chunkSize = ARRAY_SIZE / numCPUs;
	long start = participant * chunkSize;
	long end = (participant + 1L) * chunkSize;
	if (participant == numCPUs-1) {
		end = ARRAY_SIZE;
	}
	
	nanos_user_lock(&critical_handle, __FILE_LINE__);
	concurrent_tasks++;
	tap.evaluate(concurrent_tasks == 1, "Check that only one task is in the critical region after entering it");
	for (long i=start; i < end; i++) {
		sum_result += data[i];
	}
	tap.evaluate(concurrent_tasks == 1, "Check that only one task is in the critical region before exiting it");
	concurrent_tasks--;
	nanos_user_unlock(&critical_handle);
}

static void sum_wrapper(void *argsBlock)
{
	long *participantPointer = (long *) argsBlock;
	
	sum(*participantPointer);
}

static void sum_register_depinfo(__attribute__((unused)) void *handler, __attribute__((unused)) void *argsBlock)
{
}

static void sum_register_copies(__attribute__((unused)) void *handler, __attribute__((unused)) void *argsBlock)
{
}

static nanos_task_info sum_info = {
	sum_wrapper,
	sum_register_depinfo,
	sum_register_copies,
	"sum",
	"sum_source_line"
};


int main(int argc, char **argv) {
	while (!ThreadManagerDebuggingInterface::hasFinishedCPUInitialization()) {
		sched_yield();
	}
	
	numCPUs = ThreadManager::getTotalCPUs();
	
	tap.registerNewTests(numCPUs * 2L + 1);
	
	// Initialize in parallel
	data = new long[ARRAY_SIZE];
	for (int i=0; i < numCPUs ; i++) {
		long *initialize_args_block;
		void *initialize_task = nullptr;
		static nanos_task_invocation_info initialize_invocation_info = {
			__FILE_LINE__
		};
		nanos_create_task(&initialize_info, &initialize_invocation_info, sizeof(long), (void **) &initialize_args_block, &initialize_task);
		*initialize_args_block = i;
		nanos_submit_task(initialize_task);
	}
	nanos_taskwait(__FILE_LINE__);
	
	tap.begin();
	
	// Sum in tasks but with a critical inside
	concurrent_tasks = 0;
	for (int i=0; i < numCPUs ; i++) {
		long *sum_args_block;
		void *sum_task = nullptr;
		static nanos_task_invocation_info sum_invocation_info = {
			__FILE_LINE__
		};
		nanos_create_task(&sum_info, &sum_invocation_info, sizeof(long), (void **) &sum_args_block, &sum_task);
		*sum_args_block = i;
		nanos_submit_task(sum_task);
	}
	nanos_taskwait(__FILE_LINE__);
	
	tap.evaluate(sum_result == ARRAY_SIZE, "Check that the result is correct");
	tap.emitDiagnostic<>("Expected result: ", ARRAY_SIZE);
	tap.emitDiagnostic<>("Actual result: ", sum_result);
	
	return 0;
}

