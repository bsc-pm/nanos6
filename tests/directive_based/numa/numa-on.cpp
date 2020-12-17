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

	nanos6_bitmask_t bitmask;
	nanos6_bitmask_set_wildcard(&bitmask, NUMA_ALL);
	size_t numaNodes = nanos6_count_setbits(&bitmask);

	if (numaNodes == 1) {
		tap.skip("This test does not work with just 1 NUMA node");
		tap.end();
		return 0;
	}

	int enabled = nanos6_is_numa_tracking_enabled();
	tap.evaluate(
		enabled,
		"Check that NUMA tracking is enabled"
	);

	tap.bailOutAndExitIfAnyFailed();

	tap.end();

	return 0;
}
