/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
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

//! \brief Performs a memory allocation for a specific device type
void *oss_device_alloc(oss_device_t device, int index, size_t size, size_t access_stride);

//! \brief Frees a previous memory allocation
void oss_device_free(void *ptr);

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif /* NANOS6_DIRECTORY_H */
