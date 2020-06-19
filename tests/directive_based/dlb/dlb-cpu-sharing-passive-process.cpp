/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include <cstdlib>  /* getenv */
#include <cstring>  /* strcmp */
#include <string>
#include <unistd.h> /* usleep */

#include <nanos6/debug.h>

#include "TestAnyProtocolProducer.hpp"


TestAnyProtocolProducer tap;

void wrongExecution(const char *error)
{
	tap.registerNewTests(1);
	tap.begin();
	tap.success(error);
	tap.end();
}

int main(int argc, char **argv) {
	// NOTE: This test should only be ran from the dlb-cpu-sharing test
	if (argc == 1) {
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

	// Simply wait for 5 seconds, while almost all the CPUs are lent to other
	// processes, and then quit. This program is just a dummy process that
	// serves its CPUs to other running processes
	usleep(5000000);

	return 0;
}
