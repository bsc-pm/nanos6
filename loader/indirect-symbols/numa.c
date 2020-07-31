/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


#pragma GCC visibility push(default)

void *nanos6_numa_alloc_interleaved_subset(
	size_t size,
	nanos6_bitmask_t *bitmask,
	size_t block_size
) {
	typedef void *nanos6_numa_alloc_interleaved_subset(
			size_t size, nanos6_bitmask_t *bitmask, size_t block_size
	);

	static nanos6_numa_alloc_interleaved_subset *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_numa_alloc_interleaved_subset *) _nanos6_resolve_symbol("nanos6_numa_alloc_interleaved_subset", "numa", NULL);
	}

	return (*symbol)(size, bitmask, block_size);
}

void nanos6_numa_free(
	void *ptr,
	size_t size,
	nanos6_bitmask_t *bitmask,
	size_t block_size
) {
	typedef void nanos6_numa_free(
			void *ptr, size_t size, nanos6_bitmask_t *bitmask, size_t block_size
	);

	static nanos6_numa_free *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_numa_free *) _nanos6_resolve_symbol("nanos6_numa_free", "numa", NULL);
	}

	(*symbol)(ptr, size, bitmask, block_size);
}

#pragma GCC visibility pop

