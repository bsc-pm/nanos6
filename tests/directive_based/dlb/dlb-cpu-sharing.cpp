/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sched.h>
#include <sstream>
#include <sys/sysinfo.h>

#include "TestAnyProtocolProducer.hpp"


int main(int argc, char **argv) {
	char *dlbEnabled = std::getenv("NANOS6_ENABLE_DLB");
	if (dlbEnabled == 0) {
		TestAnyProtocolProducer tap;
		tap.registerNewTests(1);
		tap.begin();
		tap.success("DLB is disabled, skipping this test");
		tap.end();
		return 0;
	} else if (strcmp(dlbEnabled, "1") != 0) {
		TestAnyProtocolProducer tap;
		tap.registerNewTests(1);
		tap.begin();
		tap.success("DLB is disabled, skipping this test");
		tap.end();
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
	std::system("dlb_shm -d");

	// Invoke the first command
	std::stringstream firstCommand;
	long passiveFirstCPU = 0;
	long passiveLastCPU  = (activeCPUs / 2) - 1;
	firstCommand
		<< "taskset -c "
		<< passiveFirstCPU << "-" << passiveLastCPU << " "
		<< passiveProcessString << " nanos6-testing "
		<< passiveFirstCPU << " " << passiveLastCPU << " &";

	const std::string firstCommandString(firstCommand.str());
	std::system(firstCommandString.c_str());


	// Invoke the second command
	std::stringstream secondCommand;
	long activeFirstCPU = (activeCPUs / 2);
	long activeLastCPU  = activeCPUs - 1;
	secondCommand
		<< "taskset -c "
		<< activeFirstCPU << "-" << activeLastCPU << " "
		<< activeProcessString << " nanos6-testing "
		<< passiveFirstCPU << " " << passiveLastCPU;

	const std::string secondCommandString(secondCommand.str());
	std::system(secondCommandString.c_str());
}
