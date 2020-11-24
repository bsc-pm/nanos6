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
	void *ptr = nanos6_numa_alloc_block_interleave(32768, &bitmask, 4096);
	void *ptr_sentinel = nanos6_numa_alloc_sentinels(32768, &bitmask, 4096);
	enabled = nanos6_is_numa_tracking_enabled();
	tap.evaluate(
		enabled,
		"Check that NUMA tracking is enabled, because we already did an allocation"
	);

	nanos6_bitmask_set_wildcard(&bitmask, NUMA_ALL_ACTIVE);
	void *ptr2 = nanos6_numa_alloc_block_interleave(32768, &bitmask, 4096);
	void *ptr2_sentinel = nanos6_numa_alloc_sentinels(32768, &bitmask, 4096);

	nanos6_bitmask_set_wildcard(&bitmask, NUMA_ANY_ACTIVE);
	void *ptr3 = nanos6_numa_alloc_block_interleave(32768, &bitmask, 4096);
	void *ptr3_sentinel = nanos6_numa_alloc_sentinels(32768, &bitmask, 4096);

	tap.bailOutAndExitIfAnyFailed();

	tap.end();

	nanos6_numa_free(ptr);
	nanos6_numa_free(ptr_sentinel);
	nanos6_numa_free(ptr2);
	nanos6_numa_free(ptr2_sentinel);
	nanos6_numa_free(ptr3);
	nanos6_numa_free(ptr3_sentinel);

	return 0;
}
