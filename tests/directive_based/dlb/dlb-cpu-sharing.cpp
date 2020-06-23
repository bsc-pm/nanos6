/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/debug.h>

#include <cstdlib>       /* getenv, system */
#include <cstring>       /* strcmp */
#include <iostream>
#include <sstream>       /* stringstream */
#include <sys/sysinfo.h> /* get_nprocs */

#include "TestAnyProtocolProducer.hpp"


void wrongExecution(const char *error)
{
	TestAnyProtocolProducer tap;
	tap.registerNewTests(1);
	tap.begin();
	tap.success(error);
	tap.end();
}

int main(int argc, char **argv) {
	// Make sure DLB is enabled
	char *dlbEnabled = std::getenv("NANOS6_ENABLE_DLB");
	if (dlbEnabled == 0) {
		wrongExecution("DLB is disabled, skipping this test");
		return 0;
	} else if (strcmp(dlbEnabled, "1") != 0) {
		wrongExecution("DLB is disabled, skipping this test");
		return 0;
	}

	// Check that we're using all the CPUs in the system
	long activeCPUs = get_nprocs();
	unsigned long activeCPUsNanos6 = nanos6_get_num_cpus();
	if (activeCPUs != activeCPUsNanos6) {
		wrongExecution("Can only execute this test using all available cores, skipping this test");
		return 0;
	}

	// Make sure we're using an even number of CPUs
	if (activeCPUs % 2) {
		activeCPUs -= 1;
	}

	// Make sure we're using enough CPUs
	if (activeCPUs < 4) {
		wrongExecution("Can only execute this test with 4 or more CPUs, skipping this test");
		return 0;
	}

	// Check if this is the debug version of the program
	std::string testName(argv[0]);
	std::string activeProcessString;
	if (testName.find("debug") != std::string::npos) {
		activeProcessString  = "./dlb-cpu-sharing-active-process.debug.test";
	} else {
		activeProcessString  = "./dlb-cpu-sharing-active-process.test";
	}

	// Delete the shared memory so the subprocesses can be executed
	if (!std::system("dlb_shm -d > /dev/null 2>&1")) {
		long firstCPU = (activeCPUs / 2);
		long lastCPU = activeCPUs - 1;
		std::stringstream activeCPUs;
		activeCPUs << "taskset -c " << firstCPU << "-" << lastCPU << " " << activeProcessString << " nanos6-testing";
		std::system(activeCPUs.str().c_str());

		return 0;
	} else {
		// DLB is not found, abort and skip this test
		wrongExecution("DLB not found in $PATH, skipping this test");
		return 0;
	}
}
