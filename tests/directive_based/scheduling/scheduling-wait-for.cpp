/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
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
#include <vector>

#include "TestAnyProtocolProducer.hpp"


#define NUM_TASKS 10
#define NUM_REGULAR_TASKS 1000
#define NUM_WAITS 1000
#define TIMEOUT 500


TestAnyProtocolProducer tap;

bool checkTimeouts(double theoretical, std::vector<uint64_t> &timeouts, int numTimeouts, bool checkOutliers)
{
	assert(timeouts.size() == numTimeouts);

	// Sort the timeouts to get the median later
	std::sort(timeouts.begin(), timeouts.end());

	int numOutliers = 0;
	double sum = 0.0;
	double max = timeouts[0];
	double min = timeouts[0];

	for (int t = 0; t < numTimeouts; ++t) {
		double timeout = (double) timeouts[t];
		if (max < timeout)
			max = timeout;
		if (min > timeout)
			min = timeout;
		sum += timeout;
	}

	double median = (double) timeouts[numTimeouts / 2];
	double mean = sum / numTimeouts;
	double stdev = 0.0;
	for (int t = 0; t < numTimeouts; ++t) {
		double timeout = (double) timeouts[t];
		stdev += std::pow(timeout - mean, 2);
		if (timeout > 1.5 * theoretical)
			++numOutliers;
	}
	stdev = std::sqrt(stdev / numTimeouts);

	tap.emitDiagnostic("Task wait-for stats: mean ", mean,
		", median ", median, ", min ", min, ", max ", max, ", stdev ", stdev,
		", outliers ", numOutliers);

#ifdef LESS_TEST_THREADS
	const double meanFailFactor = 2.5;
#else
	const double meanFailFactor = 1.6;
#endif

	if (checkOutliers && numOutliers > numTimeouts * 0.1) {
		return false;
	} else if (mean < theoretical * 0.8 || mean > theoretical * 2.5) {
		return false;
	} else if (median < theoretical * 0.8 || median > theoretical * 1.2) {
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
	const int theoreticalTimeout = TIMEOUT;
	assert(numTasks > 0 && numWaits > 0 && theoreticalTimeout > 0);

	std::vector<std::vector<uint64_t> > timeouts(numTasks, std::vector<uint64_t>(numWaits, 0));

	tap.registerNewTests(
		/* Phase 1 */ numTasks
		/* Phase 2 */ + numTasks
	);
	tap.begin();

	tap.emitDiagnostic("Wait-for input: theoretical timeout ", theoreticalTimeout,
		", tasks ", numTasks, ", waits x task ", numWaits);

	/***********/
	/* WARM-UP */
	/***********/

	// Unfortunately mercurium does not support atomics
	volatile int warmupCounter = 0;

	for (int t = 0; t < numCPUs; ++t) {
		#pragma oss task shared(warmupCounter, numCPUs)
		{
			__sync_fetch_and_add(&warmupCounter, 1);

			while (warmupCounter < numCPUs) {
				usleep(100);
				__sync_synchronize();
			}
		}
	}
	#pragma oss taskwait

	/***********/
	/* PHASE 1 */
	/***********/

	tap.emitDiagnostic("*****************");
	tap.emitDiagnostic("***  PHASE 1  ***");
	tap.emitDiagnostic("***           ***");
	tap.emitDiagnostic("***  ", numTasks * 2, " tests ***");
	tap.emitDiagnostic("*****************");

	for (int t = 0; t < numTasks; ++t) {
		#pragma oss task shared(timeouts)
		{
			for (int i = 0; i < numWaits; ++i) {
				timeouts[t][i] = nanos6_wait_for(theoreticalTimeout);
			}
		}
	}
	#pragma oss taskwait

	for (int t = 0; t < numTasks; ++t) {
		bool correct = checkTimeouts((double) theoreticalTimeout, timeouts[t], numWaits, true);
		tap.evaluate(
			correct,
			"Check that the task accomplished almost all deadlines without regular tasks"
		);

		// Reset task timeouts for the next phase
		std::fill(timeouts[t].begin(), timeouts[t].end(), 0);
	}

	/***********/
	/* PHASE 2 */
	/***********/

	tap.emitDiagnostic("*****************");
	tap.emitDiagnostic("***  PHASE 2  ***");
	tap.emitDiagnostic("***           ***");
	tap.emitDiagnostic("***  ", numTasks * 2, " tests ***");
	tap.emitDiagnostic("*****************");

	for (int t = 0; t < numTasks; ++t) {
		#pragma oss task shared(timeouts)
		{
			for (int i = 0; i < numWaits; ++i) {
				timeouts[t][i] = nanos6_wait_for(theoreticalTimeout);
			}
		}
	}

	for (int t = 0; t < numRegularTasks; ++t) {
		#pragma oss task
		{
			usleep(200);
		}
	}
	#pragma oss taskwait

	for (int t = 0; t < numTasks; ++t) {
		bool correct = checkTimeouts((double) theoreticalTimeout, timeouts[t], numWaits, false);
		tap.evaluateWeak(
			correct,
			"Check that the task accomplished almost all deadlines with other regular tasks",
			"Cannot guarantee that this test works in all machines"
		);

		// Reset task timeouts for the next phase
		std::fill(timeouts[t].begin(), timeouts[t].end(), 0);
	}

	tap.end();
}
