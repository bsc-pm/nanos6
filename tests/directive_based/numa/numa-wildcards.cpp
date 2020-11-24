/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/debug.h>

#include "TestAnyProtocolProducer.hpp"

TestAnyProtocolProducer tap;


int main(int argc, char **argv) {

	nanos6_wait_for_full_initialization();

	tap.registerNewTests(3);
	tap.begin();

	nanos6_bitmask_t bitmask_all, bitmask_all_active, bitmask_any_active;
	nanos6_bitmask_set_wildcard(&bitmask_all, NUMA_ALL);
	nanos6_bitmask_set_wildcard(&bitmask_all_active, NUMA_ALL_ACTIVE);
	nanos6_bitmask_set_wildcard(&bitmask_any_active, NUMA_ANY_ACTIVE);
	size_t num_all = nanos6_count_setbits(&bitmask_all);
	size_t num_all_active = nanos6_count_setbits(&bitmask_all_active);
	size_t num_any_active = nanos6_count_setbits(&bitmask_any_active);
	tap.evaluate(
		num_all >= num_all_active,
		"Check that NUMA_ALL wildcard enables same or more bits than NUMA_ALL_ACTIVE"
	);

	tap.evaluate(
		num_all >= num_any_active,
		"Check that NUMA_ALL wildcard enables same or more bits than NUMA_ANY_ACTIVE"
	);

	tap.evaluate(
		num_any_active >= num_all_active,
		"Check that NUMA_ANY_ACTIVE wildcard enables same or more bits than NUMA_ALL_ACTIVE"
	);

	tap.bailOutAndExitIfAnyFailed();

	tap.end();

	return 0;
}
