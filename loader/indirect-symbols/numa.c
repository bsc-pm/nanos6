/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


#pragma GCC visibility push(default)

void *nanos6_numa_alloc_block_interleave(
	uint64_t size,
	const nanos6_bitmask_t *bitmask,
	uint64_t block_size
) {
	typedef void *nanos6_numa_alloc_block_interleave_t(
		uint64_t size, const nanos6_bitmask_t *bitmask, uint64_t block_size
	);

	static nanos6_numa_alloc_block_interleave_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_numa_alloc_block_interleave_t *) _nanos6_resolve_symbol("nanos6_numa_alloc_block_interleave", "numa", NULL);
	}

	return (*symbol)(size, bitmask, block_size);
}

void *nanos6_numa_alloc_sentinels(
	uint64_t size,
	const nanos6_bitmask_t *bitmask,
	uint64_t block_size
) {
	typedef void *nanos6_numa_alloc_sentinels_t(
		uint64_t size, const nanos6_bitmask_t *bitmask, uint64_t block_size
	);

	static nanos6_numa_alloc_sentinels_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_numa_alloc_sentinels_t *) _nanos6_resolve_symbol("nanos6_numa_alloc_sentinels", "numa", NULL);
	}

	return (*symbol)(size, bitmask, block_size);
}

void nanos6_numa_free(void *ptr)
{
	typedef void nanos6_numa_free_t(void *ptr);

	static nanos6_numa_free_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_numa_free_t *) _nanos6_resolve_symbol("nanos6_numa_free", "numa", NULL);
	}

	(*symbol)(ptr);
}

void nanos6_bitmask_clearall(nanos6_bitmask_t *bitmask)
{
	typedef void nanos6_bitmask_clearall_t(nanos6_bitmask_t *bitmask);

	static nanos6_bitmask_clearall_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_bitmask_clearall_t *) _nanos6_resolve_symbol("nanos6_bitmask_clearall", "numa", NULL);
	}

	(*symbol)(bitmask);
}

void nanos6_bitmask_clearbit(nanos6_bitmask_t *bitmask, uint64_t n)
{
	typedef void nanos6_bitmask_clearbit_t(nanos6_bitmask_t *bitmask, uint64_t n);

	static nanos6_bitmask_clearbit_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_bitmask_clearbit_t *) _nanos6_resolve_symbol("nanos6_bitmask_clearbit", "numa", NULL);
	}

	(*symbol)(bitmask, n);
}

void nanos6_bitmask_setall(nanos6_bitmask_t *bitmask)
{
	typedef void nanos6_bitmask_setall_t(nanos6_bitmask_t *bitmask);

	static nanos6_bitmask_setall_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_bitmask_setall_t *) _nanos6_resolve_symbol("nanos6_bitmask_setall", "numa", NULL);
	}

	(*symbol)(bitmask);
}

void nanos6_bitmask_setbit(nanos6_bitmask_t *bitmask, uint64_t n)
{
	typedef void nanos6_bitmask_setbit_t(nanos6_bitmask_t *bitmask, uint64_t n);

	static nanos6_bitmask_setbit_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_bitmask_setbit_t *) _nanos6_resolve_symbol("nanos6_bitmask_setbit", "numa", NULL);
	}

	(*symbol)(bitmask, n);
}

void nanos6_bitmask_set_wildcard(
	nanos6_bitmask_t *bitmask,
	nanos6_bitmask_wildcard_t wildcard
) {
	typedef void nanos6_bitmask_set_wildcard_t(
		nanos6_bitmask_t *bitmask, nanos6_bitmask_wildcard_t wildcard
	);

	static nanos6_bitmask_set_wildcard_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_bitmask_set_wildcard_t *) _nanos6_resolve_symbol("nanos6_bitmask_set_wildcard", "numa", NULL);
	}

	(*symbol)(bitmask, wildcard);
}

uint64_t nanos6_bitmask_isbitset(const nanos6_bitmask_t *bitmask, uint64_t n)
{
	typedef uint64_t nanos6_bitmask_isbitset_t(const nanos6_bitmask_t *bitmask, uint64_t n);

	static nanos6_bitmask_isbitset_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_bitmask_isbitset_t *) _nanos6_resolve_symbol("nanos6_bitmask_isbitset", "numa", NULL);
	}

	return (*symbol)(bitmask, n);
}

uint64_t nanos6_count_setbits(const nanos6_bitmask_t *bitmask)
{
	typedef uint64_t nanos6_count_setbits_t(const nanos6_bitmask_t *bitmask);

	static nanos6_count_setbits_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_count_setbits_t *) _nanos6_resolve_symbol("nanos6_count_setbits", "numa", NULL);
	}

	return (*symbol)(bitmask);
}

#pragma GCC visibility pop

