/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


#pragma GCC visibility push(default)

void *nanos6_numa_alloc_interleaved_subset(
	uint64_t size,
	nanos6_bitmask_t *bitmask,
	uint64_t block_size
) {
	typedef void *nanos6_numa_alloc_interleaved_subset(
			uint64_t size, nanos6_bitmask_t *bitmask, uint64_t block_size
	);

	static nanos6_numa_alloc_interleaved_subset *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_numa_alloc_interleaved_subset *) _nanos6_resolve_symbol("nanos6_numa_alloc_interleaved_subset", "numa", NULL);
	}

	return (*symbol)(size, bitmask, block_size);
}

void *nanos6_numa_alloc_sentinels(
	uint64_t size,
	nanos6_bitmask_t *bitmask,
	uint64_t block_size
) {
	typedef void *nanos6_numa_alloc_sentinels(
			uint64_t size, nanos6_bitmask_t *bitmask, uint64_t block_size
	);

	static nanos6_numa_alloc_sentinels *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_numa_alloc_sentinels *) _nanos6_resolve_symbol("nanos6_numa_alloc_sentinels", "numa", NULL);
	}

	return (*symbol)(size, bitmask, block_size);
}

void nanos6_numa_free_debug(
	void *ptr,
	uint64_t size,
	nanos6_bitmask_t *bitmask,
	uint64_t block_size
) {
	typedef void nanos6_numa_free_debug(
			void *ptr, uint64_t size, nanos6_bitmask_t *bitmask, uint64_t block_size
	);

	static nanos6_numa_free_debug *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_numa_free_debug *) _nanos6_resolve_symbol("nanos6_numa_free_debug", "numa", NULL);
	}

	(*symbol)(ptr, size, bitmask, block_size);
}

void nanos6_numa_free(
	void *ptr
) {
	typedef void nanos6_numa_free(
			void *ptr
	);

	static nanos6_numa_free *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_numa_free *) _nanos6_resolve_symbol("nanos6_numa_free", "numa", NULL);
	}

	(*symbol)(ptr);
}

void nanos6_bitmask_clearall(
	 nanos6_bitmask_t *bitmask
) {
	typedef void nanos6_bitmask_clearall(
			nanos6_bitmask_t *bitmask
	);

	static nanos6_bitmask_clearall *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_bitmask_clearall *) _nanos6_resolve_symbol("nanos6_bitmask_clearall", "numa", NULL);
	}

	(*symbol)(bitmask);
}

void nanos6_bitmask_clearbit(
	 nanos6_bitmask_t *bitmask,
	 uint64_t n
) {
	typedef void nanos6_bitmask_clearbit(
			nanos6_bitmask_t *bitmask, uint64_t n
	);

	static nanos6_bitmask_clearbit *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_bitmask_clearbit *) _nanos6_resolve_symbol("nanos6_bitmask_clearbit", "numa", NULL);
	}

	(*symbol)(bitmask, n);
}

void nanos6_bitmask_setall(
	 nanos6_bitmask_t *bitmask
) {
	typedef void nanos6_bitmask_setall(
			nanos6_bitmask_t *bitmask
	);

	static nanos6_bitmask_setall *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_bitmask_setall *) _nanos6_resolve_symbol("nanos6_bitmask_setall", "numa", NULL);
	}

	(*symbol)(bitmask);
}

void nanos6_bitmask_setbit(
	 nanos6_bitmask_t *bitmask,
	 uint64_t n
) {
	typedef void nanos6_bitmask_setbit(
			nanos6_bitmask_t *bitmask, uint64_t n
	);

	static nanos6_bitmask_setbit *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_bitmask_setbit *) _nanos6_resolve_symbol("nanos6_bitmask_setbit", "numa", NULL);
	}

	(*symbol)(bitmask, n);
}

uint8_t nanos6_bitmask_isbitset(
	 nanos6_bitmask_t *bitmask,
	 uint64_t n
) {
	typedef void nanos6_bitmask_isbitset(
			nanos6_bitmask_t *bitmask, uint64_t n
	);

	static nanos6_bitmask_isbitset *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_bitmask_isbitset *) _nanos6_resolve_symbol("nanos6_bitmask_isbitset", "numa", NULL);
	}

	return (*symbol)(bitmask, n);
}

uint8_t nanos6_get_numa_nodes(
	 nanos6_bitmask_t *bitmask
) {
	typedef void nanos6_get_numa_nodes(
			nanos6_bitmask_t *bitmask
	);

	static nanos6_get_numa_nodes *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_get_numa_nodes *) _nanos6_resolve_symbol("nanos6_get_numa_nodes", "numa", NULL);
	}

	return (*symbol)(bitmask);
}

#pragma GCC visibility pop

