/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023-2024 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_DIRECTORY_H
#define NANOS6_DIRECTORY_H

#include "major.h"

#include <unistd.h>

#pragma GCC visibility push(default)

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	oss_device_host,
	oss_device_cuda,
	oss_device_max
} oss_device_t;

typedef enum {
	//! Defines that the allocation must be contiguous even on non-home nodes
	oss_device_alloc_contiguous = (1 << 0),
} oss_device_alloc_flags_t;

//! \brief Performs a memory allocation for a specific device type
//!
//! This function will allocate a memory region in the home device, specificed by the
//! device type and the index. The resulting memory may only be accessed by tasks
//! executed outside the home device when they declare a dependency within a region of
//! the allocated memory of size access_stride.
//!
//! \param[in] device The device type of the home node
//! \param[in] index Relative index of the home node device within its type
//! \param[in] size Allocation size
//! \param[in] access_stride Size of the allocation region non-home tasks will access
//! \param[in] flags Flags of the allocation
//!
//! \returns the real address of the allocation (host) or a virtual address range representing the allocation (non-host)
void *oss_device_alloc(oss_device_t device, int index, size_t size, size_t access_stride, size_t flags);

//! \brief Frees a previous memory allocation
//!
//! \param[in] ptr Address of the allocation as returned by oss_device_alloc
void oss_device_free(void *ptr);

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif /* NANOS6_DIRECTORY_H */
