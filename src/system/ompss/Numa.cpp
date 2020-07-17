/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6.h>

#include <cassert>

#include "memory/manager-numa/ManagerNUMA.hpp"

void *nanos6_numa_alloc_interleaved_subset(
	size_t size,
	nanos6_bitmask_t *bitmask,
	size_t block_size
) {
	return ManagerNUMA::alloc(size, bitmask, block_size);
}

void nanos6_numa_free(
	void *ptr,
	size_t size
) {
	ManagerNUMA::free(ptr, size);
}

