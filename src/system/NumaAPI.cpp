/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6.h>

#include "memory/numa/NUMAManager.hpp"

void *nanos6_numa_alloc_block_interleave(
	uint64_t size,
	const nanos6_bitmask_t *bitmask,
	uint64_t block_size
) {
	return NUMAManager::alloc(size, bitmask, block_size);
}

void *nanos6_numa_alloc_sentinels(
	uint64_t size,
	const nanos6_bitmask_t *bitmask,
	uint64_t block_size
) {
	return NUMAManager::allocSentinels(size, bitmask, block_size);
}

void nanos6_numa_free(void *ptr)
{
	NUMAManager::free(ptr);
}

void nanos6_bitmask_clearall(nanos6_bitmask_t *bitmask)
{
	NUMAManager::clearAll(bitmask);
}

void nanos6_bitmask_clearbit(nanos6_bitmask_t *bitmask, uint64_t n)
{
	NUMAManager::clearBit(bitmask, n);
}

void nanos6_bitmask_setall(nanos6_bitmask_t *bitmask)
{
	NUMAManager::setAll(bitmask);
}

void nanos6_bitmask_setbit(nanos6_bitmask_t *bitmask, uint64_t n)
{
	NUMAManager::setBit(bitmask, n);
}

void nanos6_bitmask_set_wildcard(
	nanos6_bitmask_t *bitmask,
	nanos6_bitmask_wildcard_t wildcard
) {
	NUMAManager::setWildcard(bitmask, wildcard);
}

uint64_t nanos6_bitmask_isbitset(const nanos6_bitmask_t *bitmask, uint64_t n)
{
	return NUMAManager::isBitSet(bitmask, n);
}

uint64_t nanos6_count_setbits(const nanos6_bitmask_t *bitmask)
{
	return NUMAManager::countEnabledBits(bitmask);
}
