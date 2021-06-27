/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6.h>
#include <nanos6/debug.h>

#include <cassert>
#include <cstdlib>
#include <pthread.h>
#include <unistd.h>
#include <vector>

#include "Atomic.hpp"
#include "TestAnyProtocolProducer.hpp"

#define NUM_TASKS 1000

typedef std::vector<Atomic<void *> > counters_list_t;
typedef std::vector<Atomic<bool> > processed_list_t;

struct FulfillerArgs {
	counters_list_t *_counters;
	processed_list_t *_processed;

	FulfillerArgs(counters_list_t *counters, processed_list_t *processed) :
		_counters(counters),
		_processed(processed)
	{
	}
};

TestAnyProtocolProducer tap;


void *fulfiller(void *arg)
{
	FulfillerArgs *args = (FulfillerArgs *) arg;
	assert(args != NULL);

	counters_list_t &counters = *args->_counters;
	processed_list_t &processed = *args->_processed;

	// Wait until tasks can finish
	sleep(3);

	// Iterate through all counters fulfilling the
	// missing task events
	for (int c = 0; c < counters.size(); ++c) {
		void *counter = NULL;
		while (!(counter = counters[c])) {
			usleep(100);
		}
		processed[c] = true;

		if (counter != (void *) -1) {
			nanos6_decrease_task_event_counter(counter, 1);
			nanos6_decrease_task_event_counter(counter, 1);
		}
	}
	return NULL;
}

void onreadyFunction(int task, counters_list_t &counters)
{
	void *counter = nanos6_get_current_event_counter();
	assert(counter != NULL);

	nanos6_increase_current_task_event_counter(counter, 2);

	// The pre-events of even tasks will be fulfilled by an external thread
	if (task % 2 == 0) {
		counters[task] = counter;
	} else {
		nanos6_decrease_task_event_counter(counter, 1);
		nanos6_decrease_task_event_counter(counter, 1);
		counters[task] = (void *) -1;
	}
}

int main(int argc, char **argv)
{
	const long activeCPUs = nanos6_get_num_cpus();
	tap.emitDiagnostic("Detected ", activeCPUs, " CPUs");

	if (activeCPUs == 1) {
		// This test only works correctly with more than 1 CPU
		tap.registerNewTests(1);
		tap.begin();
		tap.skip("This test does not work with just 1 CPU");
		tap.end();
		return 0;
	}

	const int ntasks = NUM_TASKS;
	tap.registerNewTests(ntasks * 2);
	tap.begin();

	counters_list_t preCounters(ntasks);
	counters_list_t postCounters(ntasks);
	processed_list_t preProcessed(ntasks);
	processed_list_t postProcessed(ntasks);

	for (int c = 0; c < ntasks; ++c) {
		preCounters[c] = NULL;
		postCounters[c] = NULL;
		preProcessed[c] = false;
		postProcessed[c] = false;
	}

	// Create external threads that will fulfill the task events
	pthread_t preThread, postThread;
	FulfillerArgs preArgs(&preCounters, &preProcessed);
	FulfillerArgs postArgs(&postCounters, &postProcessed);

	int eret;
	eret = pthread_create(&preThread, NULL, fulfiller, &preArgs);
	assert(!eret);

	eret = pthread_create(&postThread, NULL, fulfiller, &postArgs);
	assert(!eret);

	for (int task = 0; task < ntasks; ++task) {
		#pragma oss task shared(postCounters, preCounters) onready(onreadyFunction(task, preCounters))
		{
			void *counter = nanos6_get_current_event_counter();
			assert(counter != NULL);

			// Increase the event counter by two and save it; the
			// fulfiller thread will decrease by two
			nanos6_increase_current_task_event_counter(counter, 2);
			postCounters[task] = counter;

			// Increase the counter by two again
			nanos6_increase_current_task_event_counter(counter, 2);

			// Decrease the counter by two; we are missing two events
			nanos6_decrease_task_event_counter(counter, 1);
			nanos6_decrease_task_event_counter(counter, 1);
		}
	}
	#pragma oss taskwait

	// Check that all counters were processed by the fulfillers
	for (int c = 0; c < ntasks; ++c) {
		tap.evaluate(preProcessed[c], "Check that the task pre-event was fulfilled");
		tap.evaluate(postProcessed[c], "Check that the task post-event was fulfilled");
	}

	eret = pthread_join(preThread, NULL);
	assert(!eret);

	eret = pthread_join(postThread, NULL);
	assert(!eret);

	tap.end();

	return 0;
}
