/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2024 Barcelona Supercomputing Center (BSC)
*/

#include <unistd.h>

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

	tap.registerNewTests(2);
	tap.begin();

	int enabled = nanos6_is_numa_tracking_enabled();
	tap.evaluate(
		!enabled,
		"Check that NUMA tracking is disabled, because there was no allocation yet"
	);

	nanos6_bitmask_set_wildcard(&bitmask, NUMA_ALL);

	int pagesize = getpagesize();
	void *ptr = nanos6_numa_alloc_block_interleave(pagesize*8, &bitmask, pagesize);
	void *ptr_sentinel = nanos6_numa_alloc_sentinels(pagesize*8, &bitmask, pagesize);
	enabled = nanos6_is_numa_tracking_enabled();
	tap.evaluate(
		enabled,
		"Check that NUMA tracking is enabled, because we already did an allocation"
	);
	nanos6_numa_free(ptr);
	nanos6_numa_free(ptr_sentinel);

	nanos6_bitmask_set_wildcard(&bitmask, NUMA_ALL_ACTIVE);

	// We may not have access to any NUMA node with all cores
	if (nanos6_count_setbits(&bitmask) > 0) {
		void *ptr2 = nanos6_numa_alloc_block_interleave(pagesize*8, &bitmask, pagesize);
		void *ptr2_sentinel = nanos6_numa_alloc_sentinels(pagesize*8, &bitmask, pagesize);
		nanos6_numa_free(ptr2);
		nanos6_numa_free(ptr2_sentinel);
	}

	nanos6_bitmask_set_wildcard(&bitmask, NUMA_ANY_ACTIVE);
	void *ptr3 = nanos6_numa_alloc_block_interleave(pagesize*8, &bitmask, pagesize);
	void *ptr3_sentinel = nanos6_numa_alloc_sentinels(pagesize*8, &bitmask, pagesize);
	nanos6_numa_free(ptr3);
	nanos6_numa_free(ptr3_sentinel);

	tap.end();

	return 0;
}
