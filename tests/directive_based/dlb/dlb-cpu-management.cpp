/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/debug.h>

#include "TestAnyProtocolProducer.hpp"
#include "Timer.hpp"

#include <cassert>
#include <sched.h>
#include <unistd.h>
#include <vector>


TestAnyProtocolProducer tap;


int main(int argc, char **argv) {
	nanos6_wait_for_full_initialization();
	
	long activeCPUs = nanos6_get_num_cpus();
	tap.emitDiagnostic("Detected ", activeCPUs, " CPUs");
	
	if (activeCPUs == 1) {
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
	
	long currentSystemCPU = nanos6_get_current_system_cpu();
	int numLentCPUs = 0;
	std::vector<bool> lentCPUs(activeCPUs, false);
	
	Timer timer;
	timer.start();
	
	// Loop until almost all active CPUs are lent
	while (numLentCPUs < (activeCPUs - 1)) {
		for (int i = 0; i < activeCPUs; ++i) {
			if (i != currentSystemCPU && !lentCPUs[i]) {
				if (nanos6_get_cpu_status(i) == nanos6_lent_cpu) {
					lentCPUs[i] = true;
					++numLentCPUs;
				}
			}
		}
		
		// Wait at most 5 seconds
		if (timer.lap() > 5000000) {
			break;
		}
	}
	
	tap.evaluate(
		numLentCPUs == (activeCPUs - 1),
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
	
	// Loop until all active CPUs are busy
	int numActiveCPUs = 0;
	while (numActiveCPUs < activeCPUs) {
		// Reset the counter
		numActiveCPUs = 0;
		
		for (int i = 0; i < activeCPUs; ++i) {
			// Keep creating work untill all CPUs are active
			#pragma oss task label(sleep)
			usleep(1000000);
			
			if (nanos6_get_cpu_status(i) != nanos6_lent_cpu) {
				++numActiveCPUs;
			}
		}
	}
	
	tap.evaluate(
		numActiveCPUs == activeCPUs,
		"Check that when enough work is available, no CPUs are lent"
	); // 4
	tap.bailOutAndExitIfAnyFailed();
	
	tap.end();
	
	return 0;
}
