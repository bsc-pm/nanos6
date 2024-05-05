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

	tap.registerNewTests(6);
	tap.begin();

	nanos6_bitmask_t bitmask;
	nanos6_bitmask_clearall(&bitmask);
	tap.evaluate(
		nanos6_count_setbits(&bitmask) == 0,
		"Check that bitmask has no bit enabled"
	);

	nanos6_bitmask_setbit(&bitmask, 0);
	tap.evaluate(
		nanos6_bitmask_isbitset(&bitmask, 0) == 1,
		"Check that recently enabled bit is enabled"
	);

	nanos6_bitmask_clearbit(&bitmask, 0);
	tap.evaluate(
		nanos6_bitmask_isbitset(&bitmask, 0) == 0,
		"Check that recently disabled bit is disabled"
	);

	nanos6_bitmask_setall(&bitmask);
	tap.evaluate(
		nanos6_count_setbits(&bitmask) > 0,
		"Check that setall enables at least 1 bit"
	);

	nanos6_bitmask_t bitmask2;
	nanos6_bitmask_set_wildcard(&bitmask2, NUMA_ALL);
	tap.evaluate(
		(bitmask == bitmask2),
		"Check that NUMA_ALL wildcard returns same result than setall"
	);

	nanos6_bitmask_set_wildcard(&bitmask2, NUMA_ANY_ACTIVE);
	tap.evaluate(
		nanos6_count_setbits(&bitmask2) > 0,
		"Check that NUMA_ANY_ACTIVE wildcard enables at least 1 bit"
	);

	tap.end();

	return 0;
}
