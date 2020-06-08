/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include <cstdlib>       /* getenv */
#include <cstring>       /* strcmp */
#include <sys/sysinfo.h> /* get_nprocs */
#include <unistd.h>      /* usleep */

#include <nanos6/debug.h>

#include "TestAnyProtocolProducer.hpp"
#include "Timer.hpp"

#include <Atomic.hpp>


#define MAX_SPINS 20000

TestAnyProtocolProducer tap;
Atomic<int> numBusyCPUs;
Atomic<int> numCheckedCPUs;


void wrongExecution(const char *error)
{
	tap.registerNewTests(1);
	tap.begin();
	tap.success(error);
	tap.end();
}

void spin()
{
	int spins = 0;
	while (spins != MAX_SPINS) {
		++spins;
	}
}


//! \brief Increases the global counter and waits untill all the CPUs in the
//! system are busy
//!
//! \param[in] numCPUs The number of CPUs used by this process (and to acquire)
void cpuComputation(long numCPUs)
{

	++numBusyCPUs;
	Timer timer;
	timer.start();

	// Wait until the number of acquired CPUs reaches numCPUs - 2
	// (minus 2 since the passive process should keep one for the
	// main task plus another for the scheduling loop)
	while (numBusyCPUs.load() < (numCPUs - 2)) {
		spin();
		// Wait for 2 seconds max
		if (timer.lap() > 2000000) {
			return;
		}
	}

	++numCheckedCPUs;
}


int main(int argc, char **argv) {
	// NOTE: This test should only be ran from the dlb-cpu-sharing test
	if (argc == 1) {
		// If there are no parameters, the program was most likely invoked
		// by autotools' make check. Skip this test without any warning
		wrongExecution("Ignoring test as it is part of a bigger one");
		return 0;
	} else if ((argc != 2) || (argc == 2 && std::string(argv[1]) != "nanos6-testing")) {
		wrongExecution("Skipping; Incorrect execution parameters");
		return 0;
	}

	char *dlbEnabled = std::getenv("NANOS6_ENABLE_DLB");
	if (dlbEnabled == 0) {
		wrongExecution("DLB is disabled, skipping this test");
		return 0;
	} else if (strcmp(dlbEnabled, "1") != 0) {
		wrongExecution("DLB is disabled, skipping this test");
		return 0;
	}

	// Retreive the current amount of CPUs
	nanos6_wait_for_full_initialization();
	size_t numCPUs = nanos6_get_num_cpus();
	tap.emitDiagnostic("Detected ", numCPUs, " CPUs");
	if (numCPUs < 4) {
		wrongExecution("Skipping; This test only works with more than 3 CPUs");
		return 0;
	}

	// ************************************************************************
	// - This test creates 1 test, and consists of one phase:
	//
	// - PHASE 1 -
	//
	// - This process only owns the first 'numCPUs/2' CPUs, thus we create numCPUs
	//   tasks and check if we can obtain all the CPUs, half of which are lent by
	//   another process.
	// - As soon as tasks are being executed, they increase the global counter of
	//   busy CPUs and wait until it reaches 'numCPUs'
	// - When the global atomic counter reaches 'numCPUs', the test has worked
	//   correctly. If it doesn't reach 'numCPUs' in a time frame of one second
	//   we account this as an expected failure
	//
	// ************************************************************************

	tap.emitDiagnostic("*********************");
	tap.emitDiagnostic("***    PHASE 1    ***");
	tap.emitDiagnostic("***               ***");
	tap.emitDiagnostic("***    1  test    ***");
	tap.emitDiagnostic("*********************");

	// Register the test
	tap.registerNewTests(1);
	tap.begin();

	// Global atomic counters
	numBusyCPUs = 0;
	numCheckedCPUs = 0;
	for (int id = 0; id < numCPUs; ++id) {
		#pragma oss task label(ownedCPUTask)
		cpuComputation(numCPUs);
	}
	#pragma oss taskwait

	// If all CPUs are acquired, success -- otherwise, do not fail the test
	// but point it out in the message
	if (numCheckedCPUs.load() == numCPUs) {
		tap.success("Check that all CPUs in the system are acquired");
	} else {
		tap.success("Check that all CPUs in the system are acquired -- weak fail as we cannot ensure it");
	}

	tap.end();

	return 0;
}
