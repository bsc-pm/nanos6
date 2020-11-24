/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_NUMA_H
#define NANOS6_NUMA_H

#include <stdint.h>

#include "major.h"


#pragma GCC visibility push(default)


// NOTE: The full version depends also on nanos6_major_api
//	   That is:   nanos6_major_api . nanos6_numa_api
enum nanos6_numa_api_t { nanos6_numa_api = 1 };

typedef enum {
	//! All the NUMA nodes available in the system
	NUMA_ALL = 0,
	//! The NUMA nodes where we have all the CPUs assigned
	NUMA_ALL_ACTIVE = 1,
	//! The NUMA nodes where we have any of the CPUs assigned
	NUMA_ANY_ACTIVE = 2
} nanos6_bitmask_wildcard_t;


#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t nanos6_bitmask_t;

//! \brief Allocate a chunk of memory distributed in different NUMA nodes
//!
//! This function allocates a chunk of memory and distributes it in the NUMA nodes specified in
//! the bitmask. The interleaving is done on a block-basis, a block is sent to a NUMA node, then,
//! the following block is sent to the next NUMA node enabled in the bitmask, and so on. A single
//! NUMA node may receive more than one block.
//!
//! \param[in] size The total size of the memory chunk to be allocated
//! \param[in] bitmask A bitmask specifying which NUMA nodes should contain a block of this chunk.
//! \param[in] block_size The block size to perform the interleaving
//!
//! \returns a pointer to the allocated memory
void *nanos6_numa_alloc_block_interleave(
	uint64_t size,
	const nanos6_bitmask_t *bitmask,
	uint64_t block_size
);

//! \brief Allocate a chunk of memory with first touch policy but annotate it in the directory as distributed
//!
//! This function allocates a chunk of memory using the default unix policy (first touch) because
//! the size of it is too small to be distributed. However we want to annotate it in the directory
//! as if it was distributed, because it is a chunk of sentinels that will be taken into account
//! for scheduling purposes, so we want it to mimic as accurately as possible the distribution of the real data
//!
//! \param[in] size The total size of the memory chunk to be allocated
//! \param[in] bitmask A bitmask specifying which NUMA nodes should contain a block of this chunk.
//! \param[in] block_size The block size to perform the interleaving
//!
//! \returns a pointer to the allocated memory
void *nanos6_numa_alloc_sentinels(
	uint64_t size,
	const nanos6_bitmask_t *bitmask,
	uint64_t block_size
);

//! \brief Deallocates a chunk of memory
//!
//! This function deallocates a chunk of memory and deallocate the corresponding entries in the directory
//!
//! \param[in] ptr The pointer to deallocate
void nanos6_numa_free(
	void *ptr
);

//! \brief Sets to 0 all the bits of the bitmask
//!
//! \param[in] bitmask The bitmask to clear all the bits
void nanos6_bitmask_clearall(
	nanos6_bitmask_t *bitmask
);

//! \brief Sets to 0 the n-th bit of the bitmask
//!
//! \param[in] bitmask The bitmask to clear the n-th bit
//! \param[in] n The position to disable
void nanos6_bitmask_clearbit(
	nanos6_bitmask_t *bitmask,
	uint64_t n
);

//! \brief Enables the N least significant bits of the bitmask, being N the number of NUMA nodes
//! available in the machine
//!
//! \param[in] bitmask The bitmask to set the bits
void nanos6_bitmask_setall(
	nanos6_bitmask_t *bitmask
);

//! \brief Sets to 1 the n-th bit of the bitmask
//!
//! \param[in] bitmask The bitmask to clear the n-th bit
//! \param[in] n The position to enable
void nanos6_bitmask_setbit(
	nanos6_bitmask_t *bitmask,
	uint64_t n
);

//! \brief Enables the bits of the bitmask corresponding to the wildcard
//!
//! \param[in] bitmask The bitmask to manipulate
//! \param[in] wildcard The wildcard to be considered
void nanos6_bitmask_set_wildcard(
	nanos6_bitmask_t *bitmask,
	nanos6_bitmask_wildcard_t wildcard
);

//! \brief Returns the value of the n-th bit of the bitmask
//!
//! \param[in] bitmask The bitmask to check the n-th bit
//! \param[in] n The position to check
//! \returns the value of the n-th bit of the bitmask
uint64_t nanos6_bitmask_isbitset(
	const nanos6_bitmask_t *bitmask,
	uint64_t n
);

//! \brief Returns the amount of enabled bits in the bitmask
//!
//! \param[in] bitmask The bitmask to count the enabled bits
//! \returns the number of enabled bits in the bitmask
uint64_t nanos6_count_setbits(
	const nanos6_bitmask_t *bitmask
);

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif /* NANOS6_NUMA_H */
