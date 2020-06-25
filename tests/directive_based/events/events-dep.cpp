/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6.h>
#include <nanos6/debug.h>

#include <cassert>
#include <cstdlib>
#include <vector>

#include "Atomic.hpp"
#include "TestAnyProtocolProducer.hpp"

#define NUM_TASKS 500
#define NUM_STEPS 500

typedef std::vector<Atomic<void*> > counters_list_t;
typedef std::vector<Atomic<bool> > processed_list_t;

TestAnyProtocolProducer tap;


void fulfill(counters_list_t &counters, processed_list_t &processed)
{
	const int ncounters = counters.size();

	// Local vector to control which ones have been fulfilled
	// Note that using the processed vector instead would add
	// overhead when accessing an Atomic variable, which may
	// be implemented with mutexes (see Atomic.hpp)
	std::vector<bool> fulfilled(ncounters, false);

	int remaining = ncounters;
	int start = 0;

	// Keep iterating and fulfilling events until all event
	// counters have been processed
	while (remaining > 0) {
		// No need to process all the vector at a time
		const int end = std::min(start + NUM_TASKS, ncounters);

		for (int c = start; c < end; ++c) {
			if (!fulfilled[c]) {
				void *counter = counters[c];
				if (counter != NULL) {
					fulfilled[c] = true;
					processed[c] = true;
					nanos6_decrease_task_event_counter(counter, 1);
					--remaining;
				}
			}
			if (fulfilled[c] && c == start) {
				++start;
			}
		}
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
	const int nsteps = NUM_STEPS;
	tap.registerNewTests(ntasks * nsteps);
	tap.begin();

	// Array for storing the test data
	int *data = (int *) calloc(ntasks, sizeof(int));
	assert(data != NULL);

	// Vector for storing all event counters
	counters_list_t counters(ntasks * nsteps);
	processed_list_t processed(ntasks * nsteps);
	for (int c = 0; c < ntasks * nsteps; ++c) {
		counters[c] = NULL;
		processed[c] = false;
	}

	// Vector for storing the exepected values at a given step
	std::vector<int> expecteds(ntasks, 0);

	// Task responsible for fulfilling all events
	#pragma oss task label(fulfill) shared(counters, processed)
	fulfill(counters, processed);

	// Initialize rand seed
	srand(time(0));

	// All tasks register an event that will be eventually fulfilled by the the
	// fulfill task. We check that the task dependencies are respected and also
	// we check that the fulfiller processed each event
	for (int step = 0; step < nsteps; ++step) {
		for (int task = 0; task < ntasks; ++task) {
			const int expected = expecteds[task];
			if (rand() % 2 == 0) {
				#pragma oss task label(reader) shared(counters, processed) in(data[task])
				{
					// Compute the task position in counters/processed
					const int current = step * ntasks + task;

					// Check only the expected value; we have no guarantee that
					// the predecessor's event was already fulfilled since the
					// reader task has an input dependency
					tap.evaluate(
						data[task] == expected,
						"Check that reads the expected value"
					);
					assert(!processed[current]);

					void *counter = nanos6_get_current_event_counter();
					assert(counter != NULL);

					// Increase and save the counter
					nanos6_increase_current_task_event_counter(counter, 1);
					counters[current] = counter;
				}
			} else {
				#pragma oss task label(writer) shared(counters, processed) inout(data[task])
				{
					const int original = data[task];
					data[task]++;

					// Compute the current/previous positions in counters/processed
					const int current = step * ntasks + task;
					const int previous = (step - 1) * ntasks + task;

					// Check the expected value and check that the predecessor's
					// event was already fulfilled; it must have been fulfilled
					// since the writer task has an inout dependency
					tap.evaluate(
						original == expected
						&& data[task] == expected + 1
						&& (previous < 0 || processed[previous]),
						"Check that reads and writes the expected value"
					);
					assert(!processed[current]);

					void *counter = nanos6_get_current_event_counter();
					assert(counter != NULL);

					// Increase and save the counter
					nanos6_increase_current_task_event_counter(counter, 1);
					counters[current] = counter;
				}
				// Only inout tasks increase the data
				++expecteds[task];
			}
		}
	}
	#pragma oss taskwait

	free(data);

	tap.end();

	return 0;
}
