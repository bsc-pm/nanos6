/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_NUMA_H
#define NANOS6_NUMA_H

#include "major.h"


#pragma GCC visibility push(default)


// NOTE: The full version depends also on nanos6_major_api
//	   That is:   nanos6_major_api . nanos6_numa_api
enum nanos6_numa_api_t { nanos6_numa_api = 1 };

//	 - NUMA_ALL: all the NUMA nodes available in the system
//	 - NUMA_ALL_ACTIVE: the NUMA nodes where we have all the CPUs assigned
//	 - NUMA_ANY_ACTIVE: the NUMA nodes where we have any of the CPUs assigned
enum nanos6_bitmask_wildcard_t {
	NUMA_ALL = 0,
	NUMA_ALL_ACTIVE = 1,
	NUMA_ANY_ACTIVE = 2
};


#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t nanos6_bitmask_t;

//! \brief Allocate a chunk of memory distributed in different NUMA nodes
//!
//! This function allocates a chunk of memory and distributes it in the NUMA nodes specified in
//! the bitmask. The interleaving is done on a block-basis, a block is sent to a NUMA node, then,
//! the following block is sent to the next NUMA node enabled in the bitmask, and so on. A single
//! NUMA node may receive more than one blocks.
//!
//! \param[in] size The total size of the memory chunk to be allocated
//! \param[in] bitmask A bitmask specifying which NUMA nodes should contain a block of this chunk.
//! \param[in] block_size The block size to perform the interleaving
void *nanos6_numa_alloc_interleaved_subset(
	uint64_t size,
	nanos6_bitmask_t *bitmask,
	uint64_t block_size
);

//! \brief Allocate a chunk of memory with first touch policy but annotate it in the directory as distributed
//!
//! This function allocates a chunk of memory using the default unix policy (first touch) because
//! the size of it is too small to be distributed. However we want to annotate it in the directory
//! as if it was distributed, because it is a chunk of sentinels that will be taken into account
//! for scheduling purposes
//!
//! \param[in] size The total size of the memory chunk to be allocated
//! \param[in] bitmask A bitmask specifying which NUMA nodes should contain a block of this chunk.
//! \param[in] block_size The block size to perform the interleaving
void *nanos6_numa_alloc_sentinels(
	uint64_t size,
	nanos6_bitmask_t *bitmask,
	uint64_t block_size
);

void nanos6_numa_free_debug(
	void *ptr,
	uint64_t size,
	nanos6_bitmask_t *bitmask,
	uint64_t block_size
);

void nanos6_numa_free(
	void *ptr
);

void nanos6_bitmask_clearall(
	nanos6_bitmask_t *bitmask
);

void nanos6_bitmask_clearbit(
	nanos6_bitmask_t *bitmask,
	uint64_t n
);

void nanos6_bitmask_setall(
	nanos6_bitmask_t *bitmask
);

void nanos6_bitmask_setbit(
	nanos6_bitmask_t *bitmask,
	uint64_t n
);

void nanos6_bitmask_set_wildcard(
	nanos6_bitmask_t *bitmask,
	nanos6_bitmask_t wildcard
);

uint8_t nanos6_bitmask_isbitset(
	nanos6_bitmask_t *bitmask,
	uint64_t n
);

uint8_t nanos6_count_enabled_bits(
	nanos6_bitmask_t *bitmask
);

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif /* NANOS6_NUMA_H */
