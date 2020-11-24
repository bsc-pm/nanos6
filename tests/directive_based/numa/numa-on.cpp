/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/debug.h>

#include "TestAnyProtocolProducer.hpp"

TestAnyProtocolProducer tap;


int main(int argc, char **argv) {

	nanos6_wait_for_full_initialization();

	tap.registerNewTests(1);
	tap.begin();

	int enabled = nanos6_is_numa_tracking_enabled();
	tap.evaluate(
		enabled,
		"Check that NUMA tracking is enabled"
	);

	tap.bailOutAndExitIfAnyFailed();

	tap.end();

	return 0;
}
