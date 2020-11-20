/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6.h>

#include "memory/manager-numa/ManagerNUMA.hpp"

void *nanos6_numa_alloc_interleaved_subset(
	uint64_t size,
	nanos6_bitmask_t *bitmask,
	uint64_t block_size
) {
	return ManagerNUMA::alloc(size, bitmask, block_size);
}

void *nanos6_numa_alloc_sentinels(
	uint64_t size,
	nanos6_bitmask_t *bitmask,
	uint64_t block_size
) {
	return ManagerNUMA::allocSentinels(size, bitmask, block_size);
}

void nanos6_numa_free_debug(
	void *ptr,
	uint64_t size,
	nanos6_bitmask_t *bitmask,
	uint64_t block_size
) {
	ManagerNUMA::freeDebug(ptr, size, bitmask, block_size);
}

void nanos6_numa_free(
	void *ptr
) {
	ManagerNUMA::free(ptr);
}

void nanos6_bitmask_clearall(
	nanos6_bitmask_t *bitmask
) {
	ManagerNUMA::clearAll(bitmask);
}

void nanos6_bitmask_clearbit(
	nanos6_bitmask_t *bitmask,
	uint64_t n
) {
	ManagerNUMA::clearBit(bitmask, n);
}

void nanos6_bitmask_setall(
	nanos6_bitmask_t *bitmask
) {
	ManagerNUMA::setAll(bitmask);
}

void nanos6_bitmask_setbit(
	nanos6_bitmask_t *bitmask,
	uint64_t n
) {
	ManagerNUMA::setBit(bitmask, n);
}

uint8_t nanos6_bitmask_isbitset(
	nanos6_bitmask_t *bitmask,
	uint64_t n
) {
	return ManagerNUMA::isBitSet(bitmask, n);
}

uint8_t nanos6_get_numa_nodes(
	nanos6_bitmask_t *bitmask
) {
	return ManagerNUMA::getNumaNodes(bitmask);
}
