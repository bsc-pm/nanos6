/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/debug.h>

#include "TestAnyProtocolProducer.hpp"

TestAnyProtocolProducer tap;


int main(int argc, char **argv) {

	nanos6_wait_for_full_initialization();

	tap.registerNewTests(2);
	tap.begin();

	int enabled = nanos6_is_numa_tracking_enabled();
	tap.evaluate(
		!enabled,
		"Check that NUMA tracking is disabled, because there was no allocation yet"
	);

	nanos6_bitmask_t bitmask;
	nanos6_bitmask_set_wildcard(&bitmask, NUMA_ALL);
	size_t numaNodes = nanos6_count_setbits(&bitmask);

	if (numaNodes == 1) {
		tap.skip("This test does not work with just 1 NUMA node");
		tap.end();
		return 0;
	}

	void *ptr = nanos6_numa_alloc_block_interleave(31496, &bitmask, 4684);
	enabled = nanos6_is_numa_tracking_enabled();
	tap.evaluate(
		enabled,
		"Check that NUMA tracking is enabled, because we already did an allocation"
	);

	nanos6_bitmask_set_wildcard(&bitmask, NUMA_ALL_ACTIVE);
	void *ptr2 = nanos6_numa_alloc_block_interleave(816349, &bitmask, 8234);

	nanos6_bitmask_set_wildcard(&bitmask, NUMA_ANY_ACTIVE);
	void *ptr3 = nanos6_numa_alloc_sentinels(9816234, &bitmask, 91824);

	tap.bailOutAndExitIfAnyFailed();

	tap.end();

	nanos6_numa_free(ptr);
	nanos6_numa_free(ptr2);
	nanos6_numa_free(ptr3);

	return 0;
}
