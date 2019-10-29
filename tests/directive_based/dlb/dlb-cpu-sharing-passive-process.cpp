/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <string>
#include <unistd.h>

#include <nanos6/debug.h>

#include "TestAnyProtocolProducer.hpp"

#include <Atomic.hpp>


#define MAX_SPINS 20000

TestAnyProtocolProducer tap;
Atomic<int> numReturnedCPUs;


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


int main(int argc, char **argv) {
	// NOTE: This test should only be ran from the dlb-cpu-sharing test
	if (argc == 1) {
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

	char *dlbEnabled = std::getenv("NANOS6_ENABLE_DLB");
	if (dlbEnabled == 0) {
		tap.registerNewTests(1);
		tap.begin();
		tap.success("DLB is disabled, skipping this test");
		tap.end();
		return 0;
	} else if (strcmp(dlbEnabled, "1") != 0) {
		tap.registerNewTests(1);
		tap.begin();
		tap.success("DLB is disabled, skipping this test");
		tap.end();
		return 0;
	}

	// Retreive the current amount of CPUs
	nanos6_wait_for_full_initialization();
	size_t numCPUs = nanos6_get_num_cpus();
	if (numCPUs < 4) {
		return 0;
	}


	// ************************************************************************
	// - This test is supposed to simply be idle to lend all its unused CPUs to
	//   another process (dlb-cpu-sharing-active-process.cpp)
	//
	// - PHASE 1 -
	//
	// - We wait for 5 seconds to let the other process use all our idle CPUs
	//
	// - This phase checks that:
	//   - Upon having finished the 5 second wait, all our idle CPUs are lent
	//
	//
	// - PHASE 2 -
	//
	// - After the 5 seconds are over, we create a task per CPU in this process
	//   Each of these tasks waits until the atomic counter reaches its desired
	//   value
	//
	// - When the atomic counter reaches 'numCPUs - 1', it means all our CPUs
	//   were returned to us
	//
	// - This phase checks that:
	//   - Upon having work, all our CPUs are returned
	// ************************************************************************

	// Global atomic counters
	numReturnedCPUs = 0;

	// Wait for 5 seconds
	usleep(5000000);

	// Check that all idle CPUs are lent
	int firstCPUId = atoi(argv[2]);
	int lastCPUId = atoi(argv[3]);
	for (int i = firstCPUId; i <= lastCPUId; ++i) {
		nanos6_cpu_status_t status = nanos6_get_cpu_status(i);
		if (status == nanos6_lent_cpu) {
			++numReturnedCPUs;
		}
	}

	// Check that all unused CPUs are lent at some point
	assert(numReturnedCPUs.load() == numCPUs - 1);

	// Check that new tasks will trigger a reclaim
	for (int id = 0; id < numCPUs - 1; ++id) {
		#pragma oss task label(reactivateCPU)
		{
			long currentCPUId = nanos6_get_current_system_cpu();
			nanos6_cpu_status_t status = nanos6_get_cpu_status(currentCPUId);

			// Check that upon having work again, CPUs are reclaimed
			assert(status == nanos6_enabled_cpu);

			// Decrease the expected amount of CPUs to be reclaimed
			--numReturnedCPUs;

			// Wait until all the expected CPUs are returned
			while (numReturnedCPUs.load() != 0) {
				spin();
			}
		}
	}

	#pragma oss taskwait

	return 0;
}
