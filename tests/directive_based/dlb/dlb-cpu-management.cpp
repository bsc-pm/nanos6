/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2023 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/debug.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <unistd.h>
#include <vector>

#include "Atomic.hpp"
#include "TestAnyProtocolProducer.hpp"
#include "Timer.hpp"


TestAnyProtocolProducer tap;


int main()
{
	nanos6_wait_for_full_initialization();
	if (!nanos6_is_dlb_enabled()) {
		tap.registerNewTests(1);
		tap.begin();
		tap.success("DLB is disabled, skipping this test");
		tap.end();
		return 0;
	}

	const long numActiveCPUs = nanos6_get_num_cpus();
	tap.emitDiagnostic("Detected ", numActiveCPUs, " CPUs");

	if (numActiveCPUs == 1) {
		// This test only works correctly with more than 1 CPU
		tap.registerNewTests(1);
		tap.begin();
		tap.skip("This test does not work with just 1 CPU");
		tap.end();
		return 0;
	}

	tap.registerNewTests(
		/* Phase 1 */ 2
		/* Phase 2 */ + 1
		/* Phase 3 */ + 1
	);
	tap.begin();


	/***********/
	/* PHASE 1 */
	/***********/

	tap.emitDiagnostic("*****************");
	tap.emitDiagnostic("***  PHASE 1  ***");
	tap.emitDiagnostic("***           ***");
	tap.emitDiagnostic("***  2 tests  ***");
	tap.emitDiagnostic("*****************");

	bool enabled = nanos6_enable_cpu(0);
	tap.evaluate(
		!enabled,
		"Check that enabling a CPU has no effect when using DLB"
	); // 1
	tap.bailOutAndExitIfAnyFailed();

	bool disabled = nanos6_disable_cpu(0);
	tap.evaluate(
		!disabled,
		"Check that disabling a CPU has no effect when using DLB"
	); // 2
	tap.bailOutAndExitIfAnyFailed();


	/***********/
	/* PHASE 2 */
	/***********/

	tap.emitDiagnostic("*****************");
	tap.emitDiagnostic("***  PHASE 2  ***");
	tap.emitDiagnostic("***           ***");
	tap.emitDiagnostic("***  1 test   ***");
	tap.emitDiagnostic("*****************");

	Timer timer;
	timer.start();

	// Loop until almost all active CPUs are lent. We take into
	// account that the current CPU cannot be lent because we
	// are running on it, but also the CPU that should be inside
	// the scheduler serving tasks
	int numLentCPUs;
	do {
		numLentCPUs = 0;
		for (void *cpuIter = nanos6_cpus_begin(); cpuIter != nanos6_cpus_end(); cpuIter = nanos6_cpus_advance(cpuIter)) {
			long cpuId = nanos6_cpus_get(cpuIter);
			if (nanos6_get_cpu_status(cpuId) == nanos6_lent_cpu) {
				++numLentCPUs;
			}
		}

		// Wait at most 5 seconds
		if (timer.lap() > 5000000) {
			break;
		}
	} while (numLentCPUs < (numActiveCPUs - 2));

	tap.evaluate(
		numLentCPUs >= (numActiveCPUs - 2),
		"Check that all unused CPUs are lent after a reasonable amount of time"
	); // 3
	tap.bailOutAndExitIfAnyFailed();


	/***********/
	/* PHASE 3 */
	/***********/

	tap.emitDiagnostic("*****************");
	tap.emitDiagnostic("***  PHASE 3  ***");
	tap.emitDiagnostic("***           ***");
	tap.emitDiagnostic("***  1 test   ***");
	tap.emitDiagnostic("*****************");

	Atomic<bool> exitCondition(false);

	// Create work for all CPUs
	for (int i = 0; i < numActiveCPUs; ++i) {
		// Block CPUs until the test finishes
		#pragma oss task shared(exitCondition) label("wait")
		while (!exitCondition.load());
	}

	timer.start();

	// Loop untill all CPUs are currently running tasks. Note
	// that the tasks are forced to busy wait until the test
	// has finished
	int numRunningCPUs;
	do {
		numRunningCPUs = 0;

		// Count how many CPUs are running
		for (void *cpuIter = nanos6_cpus_begin(); cpuIter != nanos6_cpus_end(); cpuIter = nanos6_cpus_advance(cpuIter)) {
			long cpuId = nanos6_cpus_get(cpuIter);
			if (nanos6_get_cpu_status(cpuId) != nanos6_lent_cpu) {
				++numRunningCPUs;
			}
		}

		// Wait at most 5 seconds
		if (timer.lap() > 5000000) {
			break;
		}
	} while (numRunningCPUs < numActiveCPUs);

	exitCondition.store(true);

	#pragma oss taskwait

	tap.evaluate(
		numRunningCPUs == numActiveCPUs,
		"Check that when enough work is available, no CPUs are lent"
	); // 4
	tap.bailOutAndExitIfAnyFailed();

	tap.end();

	return 0;
}
