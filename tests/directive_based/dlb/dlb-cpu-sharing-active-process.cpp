/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sched.h>
#include <string>
#include <unistd.h>
#include <vector>

#include <nanos6/debug.h>

#include "TestAnyProtocolProducer.hpp"

#include <Atomic.hpp>


#define MAX_SPINS 20000

TestAnyProtocolProducer tap;
Atomic<int> numAcquiredCPUs;


void wrongExecution(const char *error)
{
	tap.registerNewTests(1);
	tap.begin();
	tap.skip(error);
	tap.end();
}

void spin()
{
	int spins = 0;
	while (spins != MAX_SPINS) {
		++spins;
	}
}

//! \param[in] numCPUs The number of CPUs used by this process (and to acquire)
void acquiredCPUComputation(long numCPUs)
{
	//    PHASE 2    //
	
	long currentCPUId = nanos6_get_current_system_cpu();
	nanos6_cpu_status_t status = nanos6_get_cpu_status(currentCPUId);
	if (status == nanos6_enabled_cpu) {
		tap.success("Check that a task that shouldn't execute in an acquired CPU is not doing so");
		
		// If the CPU is owned, halt until acquirable CPUs can execute
		// tasks (otherwise this CPU would continue executing tasks)
		while (numAcquiredCPUs.load() < numCPUs) {
			spin();
		}
	} else {
		++numAcquiredCPUs;
		tap.evaluate(
			status == nanos6_acquired_enabled_cpu,
			"Check that a task that should execute in an acquired CPU is doing so"
		);
		
		// Wait until all the CPUs that had to be acquired are acquired
		while (numAcquiredCPUs.load() < numCPUs) {
			spin();
		}
	}
}

//! \brief This function (that should be executed as a task) checks that
//! if enough work is created, a specific CPU will be acquired and then
//! returned when no more work is available
//!
//! \param[in] numCPUs The number of CPUs used by this process (and to acquire)
void ownedCPUComputation(long numCPUs)
{
	//    PHASE 1    //
	
	long currentCPUId = nanos6_get_current_system_cpu();
	tap.emitDiagnostic(
		"Task executing in owned CPU ", currentCPUId,
		", waiting until an external CPU is acquired"
	);
	
	// Wait until the number of acquired CPUs reaches numCPUs
	while (numAcquiredCPUs.load() < numCPUs) {
		spin();
	}
	
	std::ostringstream oss;
	oss << "Check that a task executing in owned CPU "
		<< currentCPUId
		<< " detects the acquire of an external CPU";
	tap.success(oss.str()); // Phase 1
}


int main(int argc, char **argv) {
	// NOTE: This test should only be ran from the dlb-cpu-sharing test
	if (argc == 1) {
		// If there are no parameters, the program was most likely invoked
		// by autotools' make check. Skip this test without any warning
		tap.registerNewTests(1);
		tap.begin();
		tap.success("Ignoring test as it is part of a bigger one");
		tap.end();
		return 0;
	} else if (
		(argc != 4) ||
		(argc == 4 && atoi(argv[2]) >= atoi(argv[3])) ||
		(argc == 4 && std::string(argv[1]) != "nanos6-testing")
	) {
		wrongExecution("Skipping; Incorrect execution parameters");
		return 0;
	}
	
	// Retreive the current amount of CPUs
	nanos6_wait_for_full_initialization();
	size_t numCPUs = nanos6_get_num_cpus();
	tap.emitDiagnostic("Detected ", numCPUs, " CPUs");
	if (numCPUs < 4 ) {
		wrongExecution("Skipping; This test only works with more than 3 CPUs");
		return 0;
	}
	
	// Make sure both processes have the same amount of CPUs
	assert((atoi(argv[3]) - atoi(argv[2])) == (numCPUs - 1));
	
	
	// ************************************************************************
	// - This test creates (numCPUs - 1) + (numCPUs * 2) tasks, and consists of
	//
	// - PHASE 1 -
	// 
	// - The first 'numCPUs - 1' tasks will be executed by CPUs owned by this
	//   process (so, one task per CPU this process has, minus one)
	// - These tasks wait until a CPU from another process is acquired
	// - When the CPU is acquired, it increases an atomic counter and waits
	//   until the counter reaches 'numCPUs - 1' (-1 since the other process
	//   needs at least one CPU)
	// - When the counter reaches 'numCPUs - 1', it means all the CPUs that
	//   could be, were acquired
	//
	// - Meanwhile, the 'numCPUs * 2' tasks will increase and wait until the
	//   counter reaches 'numCPUs - 1', so we make sure that almost all CPUs
	//   are acquired
	// - At that point, all of these tasks should be executing on acquired CPUs
	// 
	// - This phase checks that:
	//   - CPUs are acquired if all CPUs are busy and other tasks exist
	// 
	//
	// - PHASE 2 -
	//
	// - When this second half of tasks sees that the counter has reached value
	//   'numCPUs - 1', they will end their body, meaning the CPUs acquired
	//   from the other process should be returned
	//
	// - Meanwhile, the first set of tasks will be checking if their specific
	//   CPU has been returned. When that happens, the first set of tasks will
	//   reach completion
	//
	// - This phase checks that:
	//   - All acquired CPUs are returned when no more work is available
	// ************************************************************************
	
	tap.emitDiagnostic("*********************");
	tap.emitDiagnostic("***    PHASE 1    ***");
	tap.emitDiagnostic("***               ***");
	tap.emitDiagnostic("***    ", numCPUs - 1, " tests   ***");
	tap.emitDiagnostic("*********************");
	
	// Register the tests for both phases
	tap.registerNewTests(
		(numCPUs - 1) + /* Phase 1 */
		(numCPUs * 2)   /* Phase 2 */
	);
	tap.begin();
	
	// Create the counter of CPU Ids
	int firstCPUId = atoi(argv[2]);
	int lastCPUId = atoi(argv[3]);
	
	// Global atomic counter
	numAcquiredCPUs = 0;
	
	for (int id = 0; id < numCPUs - 1; ++id) {
		#pragma oss task label(ownedCPUTask)
		ownedCPUComputation(numCPUs - 1);
	}
	
	// Halt for a second so that all the owned CPU tasks can be obtained
	// by owned CPUs
	usleep(1000000);
	
	tap.emitDiagnostic("*********************");
	tap.emitDiagnostic("***    PHASE 2    ***");
	tap.emitDiagnostic("***               ***");
	tap.emitDiagnostic("***    ", numCPUs * 2, " tests   ***");
	tap.emitDiagnostic("*********************");
	
	for (int id = 0; id < numCPUs * 2; ++id) {
		#pragma oss task label(acquiredCPUTask)
		acquiredCPUComputation(numCPUs - 1);
	}
	#pragma oss taskwait
	
	tap.end();
	
	return 0;
}
