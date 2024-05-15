/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2024 Barcelona Supercomputing Center (BSC)
*/

#include <directory.h>

#include "resolve.h"

#pragma GCC visibility push(default)

void *oss_device_alloc(oss_device_t device, int index, size_t size, size_t access_stride, size_t flags)
{
	typedef void *oss_device_alloc_t(oss_device_t, int, size_t, size_t, size_t);

	static oss_device_alloc_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (oss_device_alloc_t *)
			_nanos6_resolve_symbol("oss_device_alloc", "directory", NULL);
	}

	return (*symbol)(device, index, size, access_stride, flags);
}

void oss_device_free(void *ptr)
{
	typedef void *oss_device_free_t(void *);

	static oss_device_free_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (oss_device_free_t *)
			_nanos6_resolve_symbol("oss_device_free", "directory", NULL);
	}

	return (*symbol)(ptr);
}

#pragma GCC visibility pop
