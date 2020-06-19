/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sched.h>
#include <sstream>
#include <sys/sysinfo.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

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
	char *dlbEnabled = std::getenv("NANOS6_ENABLE_DLB");
	if (dlbEnabled == 0) {
		wrongExecution("DLB is disabled, skipping this test");
		return 0;
	} else if (strcmp(dlbEnabled, "1") != 0) {
		wrongExecution("DLB is disabled, skipping this test");
		return 0;
	}

	long activeCPUs = get_nprocs();
	if (activeCPUs % 2) {
		activeCPUs -= 1;
	}

	if (activeCPUs < 4) {
		// Skip test, this test only works with more than 3 CPUs
		return 0;
	}

	// Check if this is the debug version of the program
	std::string testName(argv[0]);
	std::string activeProcessString;
	std::string passiveProcessString;
	if (testName.find("debug") != std::string::npos) {
		activeProcessString  = "./dlb-cpu-sharing-active-process.debug.test";
		passiveProcessString = "./dlb-cpu-sharing-passive-process.debug.test";
	} else {
		activeProcessString  = "./dlb-cpu-sharing-active-process.test";
		passiveProcessString = "./dlb-cpu-sharing-passive-process.test";
	}

	// Delete the shared memory so the subprocesses can be executed
	if (!std::system("dlb_shm -d > /dev/null 2>&1")) {
		// Prepare the passive command
		long passiveFirstCPU = 0;
		long passiveLastCPU = (activeCPUs / 2) - 1;
		std::stringstream passiveCommand;
		passiveCommand
			<< "taskset -c "
			<< passiveFirstCPU << "-" << passiveLastCPU << " "
			<< passiveProcessString << " nanos6-testing";
		const std::string passiveCommandString(passiveCommand.str());

		// Prepare the active command
		long activeFirstCPU = (activeCPUs / 2);
		long activeLastCPU = activeCPUs - 1;
		std::stringstream activeCommand;
		activeCommand
			<< "taskset -c "
			<< activeFirstCPU << "-" << activeLastCPU << " "
			<< activeProcessString << " nanos6-testing";
		const std::string activeCommandString(activeCommand.str());

		// Launch the active process
		#pragma oss task
		std::system(activeCommandString.c_str());

		// Launch the passive process
		#pragma oss task
		std::system(passiveCommandString.c_str());

		#pragma oss taskwait
		return 0;
	} else {
		// DLB is not found, abort and skip this test
		wrongExecution("DLB not found in $PATH, skipping this test");
		return 0;
	}
}
