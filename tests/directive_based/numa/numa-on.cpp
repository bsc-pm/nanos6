/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2024 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/debug.h>

#include "TestAnyProtocolProducer.hpp"

TestAnyProtocolProducer tap;


int main()
{
	nanos6_wait_for_full_initialization();

	nanos6_bitmask_t bitmask;
	nanos6_bitmask_set_wildcard(&bitmask, NUMA_ANY_ACTIVE);

	if (nanos6_count_setbits(&bitmask) == 1) {
		tap.registerNewTests(1);
		tap.begin();
		tap.skip("This test does not work with just 1 active NUMA node");
		tap.end();
		return 0;
	}

	tap.registerNewTests(1);
	tap.begin();

	nanos6_bitmask_set_wildcard(&bitmask, NUMA_ALL);

	int enabled = nanos6_is_numa_tracking_enabled();
	tap.evaluate(
		enabled,
		"Check that NUMA tracking is enabled"
	);

	tap.end();

	return 0;
}
