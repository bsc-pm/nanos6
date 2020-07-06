/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "TestAnyProtocolProducer.hpp"
#include "Timer.hpp"


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
		// If there are no parameters, the program was most likely invoked
		// by autotools' make check. Skip this test without any warning
		wrongExecution("Ignoring test as it is part of a bigger one");
		return 0;
	} else if ((argc != 2) || (argc == 2 && std::string(argv[1]) != "nanos6-testing")) {
		wrongExecution("Skipping; Incorrect execution parameters");
		return 0;
	}

	// This process will be killed by the active one, simply
	// wait and lend all CPUs. Ultimately, end the process
	// after 10 seconds if it still has not finished

	Timer timer;
	timer.start();
	while (1) {
		// Wait for 10 seconds max
		if (timer.lap() > 10000000) {
			return 0;
		}
	}
}
