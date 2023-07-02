/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2023 Barcelona Supercomputing Center (BSC)
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

	// This process simply waits and lends all its CPUs. Ultimately, the process
	// will end after 6

	Timer timer;
	timer.start();
	while (1) {
		if (timer.lap() > 6000000) {
			return 0;
		}
	}
}
