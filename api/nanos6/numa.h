/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_NUMA_H
#define NANOS6_NUMA_H

#include "major.h"


#pragma GCC visibility push(default)


// NOTE: The full version depends also on nanos6_major_api
//       That is:   nanos6_major_api . nanos6_numa_api
enum nanos6_numa_api_t { nanos6_numa_api = 1 };


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
	size_t size,
	nanos6_bitmask_t *bitmask,
	size_t block_size
);

void nanos6_numa_free(
	void *ptr,
	size_t size
);

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif /* NANOS6_NUMA_H */
