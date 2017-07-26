/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_MALLOC_H
#define NANOS6_MALLOC_H


#include <stddef.h>


#ifdef __cplusplus
extern "C" {
#endif

void *nanos6_intercepted_malloc(size_t size);
void nanos6_intercepted_free(void *ptr);
void *nanos6_intercepted_calloc(size_t nmemb, size_t size);
void *nanos6_intercepted_realloc(void *ptr, size_t size);
void *nanos6_intercepted_reallocarray(void *ptr, size_t nmemb, size_t size);

int nanos6_intercepted_posix_memalign(void **memptr, size_t alignment, size_t size);
void *nanos6_intercepted_aligned_alloc(size_t alignment, size_t size);
void *nanos6_intercepted_valloc(size_t size);

void *nanos6_intercepted_memalign(size_t alignment, size_t size);
void *nanos6_intercepted_pvalloc(size_t size);

#ifdef __cplusplus
}
#endif


#endif /* NANOS6_MALLOC_H */
