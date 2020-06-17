/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/blocking.h>
#include <nanos6/debug.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <unistd.h>

#include "TestAnyProtocolProducer.hpp"


#define NUM_TASKS 10
#define NUM_REGULAR_TASKS 1000
#define NUM_WAITS 1000
#define TIMEOUT 500


TestAnyProtocolProducer tap;

bool checkTimeouts(uint64_t theoric, uint64_t *timeouts, int numTimeouts, bool checkOutliers)
{
	int numOutliers = 0;
	double sum = 0.0;
	double max = timeouts[0];
	double min = timeouts[0];

	for (int t = 0; t < numTimeouts; ++t) {
		if (max < timeouts[t])
			max = (double) timeouts[t];
		if (min > timeouts[t])
			min = (double) timeouts[t];
		sum += timeouts[t];
	}

	double mean = sum / numTimeouts;
	double stdev = 0.0;
	for (int t = 0; t < numTimeouts; ++t) {
		stdev += std::pow(timeouts[t] - mean, 2);
		if (timeouts[t] > 1.5*theoric)
			++numOutliers;
	}
	stdev = std::sqrt(stdev / numTimeouts);

	tap.emitDiagnostic("Task wait-for stats: mean ", mean,
		", min ", min, ", max ", max, ", stdev ", stdev,
		", outliers ", numOutliers);

	if (checkOutliers && numOutliers > numTimeouts * 0.1) {
		return false;
	} else if (mean < theoric * 0.8 || mean > theoric * 1.4) {
		return false;
	}
	return true;
}

int main(int argc, char **argv)
{
	const int numCPUs = nanos6_get_num_cpus();

	const int numTasks = std::min(NUM_TASKS, numCPUs);
	const int numRegularTasks = NUM_REGULAR_TASKS;
	const int numWaits = NUM_WAITS;
	const int timeout = TIMEOUT;
	assert(numTasks > 0 && numWaits > 0 && timeout > 0);

	const size_t timeoutsSize = numTasks * numWaits * sizeof(uint64_t);
	uint64_t *const allTimeouts = (uint64_t *) std::malloc(timeoutsSize);
	assert(allTimeouts != NULL);

	std::memset(allTimeouts, 0, timeoutsSize);

	tap.registerNewTests(
		/* Phase 1 */ numTasks
		/* Phase 2 */ + numTasks
	);
	tap.begin();

	tap.emitDiagnostic("Wait-for input: theoric timeout ", timeout,
		", tasks ", numTasks, ", waits x task ", numWaits);

	/***********/
	/* PHASE 1 */
	/***********/

	tap.emitDiagnostic("*****************");
	tap.emitDiagnostic("***  PHASE 1  ***");
	tap.emitDiagnostic("***           ***");
	tap.emitDiagnostic("***  ", numTasks * 2, " tests ***");
	tap.emitDiagnostic("*****************");

	uint64_t *timeouts = allTimeouts;
	for (int t = 0; t < numTasks; ++t) {
		#pragma oss task
		{
			for (int i = 0; i < numWaits; ++i) {
				timeouts[i] = nanos6_wait_for(timeout);
			}
		}
		timeouts += numWaits;
	}
	#pragma oss taskwait

	timeouts = allTimeouts;
	for (int t = 0; t < numTasks; ++t) {
		bool correct = checkTimeouts(timeout, timeouts, numWaits, true);
		tap.evaluate(
			correct,
			"Check that the task accomplished almost all deadlines without regular tasks"
		);
		timeouts += numWaits;
	}
	std::memset(allTimeouts, 0, timeoutsSize);

	/***********/
	/* PHASE 2 */
	/***********/

	tap.emitDiagnostic("*****************");
	tap.emitDiagnostic("***  PHASE 2  ***");
	tap.emitDiagnostic("***           ***");
	tap.emitDiagnostic("***  ", numTasks * 2, " tests ***");
	tap.emitDiagnostic("*****************");

	timeouts = allTimeouts;
	for (int t = 0; t < numTasks; ++t) {
		#pragma oss task
		{
			for (int i = 0; i < numWaits; ++i) {
				timeouts[i] = nanos6_wait_for(timeout);
			}
		}
		timeouts += numWaits;
	}

	for (int t = 0; t < numRegularTasks; ++t) {
		#pragma oss task
		{
			usleep(200);
		}
	}
	#pragma oss taskwait

	timeouts = allTimeouts;
	for (int t = 0; t < numTasks; ++t) {
		bool correct = checkTimeouts(timeout, timeouts, numWaits, false);
		tap.evaluateWeak(
			correct,
			"Check that the task accomplished almost all deadlines with other regular tasks",
			"Cannot guarantee that this test works in all machines"
		);
		timeouts += numWaits;
	}

	std::free(allTimeouts);

	tap.end();
}
