/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include <cstdlib>  /* getenv */
#include <cstring>  /* strcmp */
#include <string>
#include <unistd.h> /* usleep */

#include <nanos6/debug.h>

#include "TestAnyProtocolProducer.hpp"


#define MAX_SPINS 20000

TestAnyProtocolProducer tap;


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

	// Wait for 5 seconds
	usleep(5000000);

	return 0;
}
