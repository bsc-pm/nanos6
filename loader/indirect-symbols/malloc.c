/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "error.h"
#include "resolve.h"

#include <dlfcn.h>
#include <errno.h>
#include <link.h>
#include <stdbool.h>
#include <stddef.h>

#ifndef HAVE_CONFIG_H
#include <config.h>
#endif


#pragma GCC visibility push(default)

// The following functions have strong __libc_ counterparts that we can use during initialization,
// since dlopen and dlsym also perform memory allocations
DECLARE_LIBC_FALLBACK(__libc_, malloc, void *, size_t);
DECLARE_LIBC_FALLBACK(__libc_, free, void, void *);
DECLARE_LIBC_FALLBACK(__libc_, calloc, void *, size_t, size_t);
DECLARE_LIBC_FALLBACK(__libc_, realloc, void *, void *, size_t);
DECLARE_LIBC_FALLBACK(__libc_, valloc, void *, size_t);
DECLARE_LIBC_FALLBACK(__libc_, memalign, void *, size_t, size_t);
DECLARE_LIBC_FALLBACK(__libc_, pvalloc, void *, size_t);

#pragma GCC visibility pop


static DECLARE_INTERCEPTED_FUNCTION_POINTER(malloc_symbol, malloc, void *, size_t) = __libc_malloc;
static DECLARE_INTERCEPTED_FUNCTION_POINTER(free_symbol, free, void, void *) = __libc_free;
static DECLARE_INTERCEPTED_FUNCTION_POINTER(calloc_symbol, calloc, void *, size_t, size_t) = __libc_calloc;
static DECLARE_INTERCEPTED_FUNCTION_POINTER(realloc_symbol, realloc, void *, void *, size_t) = __libc_realloc;
static DECLARE_INTERCEPTED_FUNCTION_POINTER(valloc_symbol, valloc, void *, size_t) = __libc_valloc;
static DECLARE_INTERCEPTED_FUNCTION_POINTER(memalign_symbol, memalign, void *, size_t, size_t) = __libc_memalign;
static DECLARE_INTERCEPTED_FUNCTION_POINTER(pvalloc_symbol, pvalloc, void *, size_t) = __libc_pvalloc;

static DECLARE_INTERCEPTED_FUNCTION_POINTER(posix_memalign_symbol, posix_memalign, int, void **, size_t, size_t) = NULL;
static DECLARE_INTERCEPTED_FUNCTION_POINTER(posix_memalign_libc_symbol, posix_memalign, int, void **, size_t, size_t) = NULL;

#if HAVE_ALIGNED_ALLOC
static DECLARE_INTERCEPTED_FUNCTION_POINTER(aligned_alloc_symbol, aligned_alloc, void *, size_t, size_t) = NULL;
static DECLARE_INTERCEPTED_FUNCTION_POINTER(aligned_alloc_libc_symbol, aligned_alloc, void *, size_t, size_t) = NULL;
#endif
#if HAVE_REALLOCARRAY
static DECLARE_INTERCEPTED_FUNCTION_POINTER(reallocarray_symbol, reallocarray, void *, void *, size_t, size_t) = NULL;
static DECLARE_INTERCEPTED_FUNCTION_POINTER(reallocarray_libc_symbol, reallocarray, void *, void *, size_t, size_t) = NULL;
#endif

#pragma GCC visibility push(default)


void *malloc(size_t size)
{ return (*malloc_symbol)(size); }

void free(void *ptr)
{ (*free_symbol)(ptr); }

void *calloc(size_t nmemb, size_t size)
{ return (*calloc_symbol)(nmemb, size); }

void *realloc(void *ptr, size_t size)
{ return (*realloc_symbol)(ptr, size); }

void *valloc(size_t size)
{ return (*valloc_symbol)(size); }

void *memalign(size_t alignment, size_t size)
{ return (*memalign_symbol)(alignment, size); }

void *pvalloc(size_t size)
{ return (*pvalloc_symbol)(size); }


typedef struct {
	char const *name;
	void *ourFunction;
	void *result;
	bool foundOurFunction;
} next_function_lookup_info_t;


static int nanos6_find_next_function_iterator(
	struct dl_phdr_info *info, size_t size, void *data
) {
	next_function_lookup_info_t *lookupInfo = (next_function_lookup_info_t *) data;
	
	if (lookupInfo->result != NULL) {
		// We already have a result
	} else {
		void *handle = dlopen(info->dlpi_name, RTLD_LAZY | RTLD_LOCAL);
		if (handle != NULL) {
			void *current = dlsym(handle, lookupInfo->name);
			if (current != NULL) {
				if (current == lookupInfo->ourFunction) {
					lookupInfo->foundOurFunction = true;
				} else if (lookupInfo->foundOurFunction) {
					lookupInfo->result = current;
				} else {
					// Skip
				}
			}
			dlclose(handle);
		} else {
			fprintf(stderr, "Warning: Could not load '%s' to look up symbol '%s'\n", info->dlpi_name, lookupInfo->name);
		}
	}
	
	return 0;
}


static int nanos6_find_next_function_error_tracer(
	struct dl_phdr_info *info, size_t size, void *data
) {
	next_function_lookup_info_t *lookupInfo = (next_function_lookup_info_t *) data;
	
	if (lookupInfo->result != NULL) {
		// We already have a result
	} else {
		fprintf(stderr, "\tChecking in '%s'\n", info->dlpi_name);
		void *handle = dlopen(info->dlpi_name, RTLD_LAZY | RTLD_LOCAL);
		if (handle != NULL) {
			void *current = dlsym(handle, lookupInfo->name);
			if (current != NULL) {
				if (lookupInfo->foundOurFunction) {
					fprintf(stderr, "\t\tFound '%s' after our own version\n", lookupInfo->name);
					lookupInfo->result = current;
				} else if (current == lookupInfo->ourFunction) {
					fprintf(stderr, "\t\tFound our own version of '%s'\n", lookupInfo->name);
					lookupInfo->foundOurFunction = true;
				} else {
					fprintf(stderr, "\t\tFound '%s' before our own version, so we are skipping it\n", lookupInfo->name);
				}
			} else {
				fprintf(stderr, "\t\tDid not find '%s' in this library\n", lookupInfo->name);
			}
			dlclose(handle);
		} else {
			fprintf(stderr, "\t\tCould not load '%s' to look up symbol '%s'\n", info->dlpi_name, lookupInfo->name);
		}
	}
	
	return 0;
}


static void *nanos6_loader_find_next_function(void *ourFunction, char const *name, bool silentFailure)
{
	next_function_lookup_info_t nextFunctionLookup = { name, ourFunction, NULL, false };
	__attribute__((unused)) int rc = dl_iterate_phdr(nanos6_find_next_function_iterator, (void *) &nextFunctionLookup);
	
	if (!silentFailure && (nextFunctionLookup.result == NULL)) {
		fprintf(stderr, "Error resolving '%s'. Lookup trace follows:\n", name);
		nextFunctionLookup.foundOurFunction = false;
		rc = dl_iterate_phdr(nanos6_find_next_function_error_tracer, (void *) &nextFunctionLookup);
		handle_error();
		return NULL;
	}
	
	return nextFunctionLookup.result;
}


int posix_memalign(void **memptr, size_t alignment, size_t size)
{
	typedef int (*posix_memalign_t)(void **, size_t, size_t);
	
	if (posix_memalign_symbol == NULL) {
		posix_memalign_libc_symbol = (posix_memalign_t) nanos6_loader_find_next_function((void *) posix_memalign, "posix_memalign", false);
		if (posix_memalign_libc_symbol == NULL) {
			fprintf(stderr, "Error resolving 'posix_memalign': %s\n", dlerror());
			handle_error();
			return EINVAL;
		}
		
		posix_memalign_symbol = posix_memalign_libc_symbol;
		REDIRECT_INTERCEPTED_FUNCTION(posix_memalign_symbol, posix_memalign, int, void **, size_t, size_t);
	}
	
	return (posix_memalign_symbol)(memptr, alignment, size);
}


#if HAVE_ALIGNED_ALLOC
void *aligned_alloc(size_t alignment, size_t size)
{
	typedef void *(*aligned_alloc_t)(size_t, size_t);
	
	if (aligned_alloc_symbol == NULL) {
		aligned_alloc_libc_symbol = (aligned_alloc_t) nanos6_loader_find_next_function((void *) aligned_alloc, "aligned_alloc", false);
		if (aligned_alloc_libc_symbol == NULL) {
			fprintf(stderr, "Error resolving 'aligned_alloc': %s\n", dlerror());
			handle_error();
			return NULL;
		}
		
		aligned_alloc_symbol = aligned_alloc_libc_symbol;
		REDIRECT_INTERCEPTED_FUNCTION(aligned_alloc_symbol, aligned_alloc, void *, size_t, size_t);
	}
	
	return (aligned_alloc_symbol)(alignment, size);
}
#endif


#if HAVE_REALLOCARRAY
void *reallocarray(void *ptr, size_t nmemb, size_t size)
{
	typedef void *(*reallocarray_t)(void *, size_t, size_t);
	
	if (reallocarray_symbol == NULL) {
		reallocarray_libc_symbol = (reallocarray_t) nanos6_loader_find_next_function((void *) reallocarray, "reallocarray", false);
		if (reallocarray_libc_symbol == NULL) {
			fprintf(stderr, "Error resolving 'reallocarray': %s\n", dlerror());
			handle_error();
			return NULL;
		}
		
		reallocarray_symbol = reallocarray_libc_symbol;
		REDIRECT_INTERCEPTED_FUNCTION(reallocarray_symbol, reallocarray, void *, void *, size_t, size_t);
	}
	
	return (reallocarray_symbol)(ptr, nmemb, size);
}
#endif


void nanos6_start_function_interception()
{
	typedef int (*posix_memalign_t)(void **, size_t, size_t);
	if (posix_memalign_symbol == NULL) {
		posix_memalign_libc_symbol = (posix_memalign_t) nanos6_loader_find_next_function((void *) posix_memalign, "posix_memalign", false);
		if (posix_memalign_libc_symbol == NULL) {
			fprintf(stderr, "Error resolving 'posix_memalign': %s\n", dlerror());
			handle_error();
		}
		
		posix_memalign_symbol = posix_memalign_libc_symbol;
	}
	
#if HAVE_ALIGNED_ALLOC
	typedef void *(*aligned_alloc_t)(size_t, size_t);
	if (aligned_alloc_symbol == NULL) {
		aligned_alloc_libc_symbol = (aligned_alloc_t) nanos6_loader_find_next_function((void *) aligned_alloc, "aligned_alloc", false);
		if (aligned_alloc_libc_symbol == NULL) {
			fprintf(stderr, "Error resolving 'aligned_alloc': %s\n", dlerror());
			handle_error();
		}
		
		aligned_alloc_symbol = aligned_alloc_libc_symbol;
	}
#endif
	
#if HAVE_REALLOCARRAY
	typedef void *(*reallocarray_t)(void *, size_t, size_t);
	if (reallocarray_symbol == NULL) {
		reallocarray_libc_symbol = (reallocarray_t) nanos6_loader_find_next_function((void *) reallocarray, "reallocarray", false);
		if (reallocarray_libc_symbol == NULL) {
			fprintf(stderr, "Error resolving 'reallocarray': %s\n", dlerror());
			handle_error();
		}
		
		reallocarray_symbol = reallocarray_libc_symbol;
	}
#endif

#if HAVE_ALIGNED_ALLOC
	REDIRECT_INTERCEPTED_FUNCTION(aligned_alloc_symbol, aligned_alloc, void *, size_t, size_t);
#endif
	
#if HAVE_REALLOCARRAY
	REDIRECT_INTERCEPTED_FUNCTION(reallocarray_symbol, reallocarray, void *, void *, size_t, size_t);
#endif
	
	REDIRECT_INTERCEPTED_FUNCTION(posix_memalign_symbol, posix_memalign, int, void **, size_t, size_t);
	
	REDIRECT_INTERCEPTED_FUNCTION(malloc_symbol, malloc, void *, size_t);
	REDIRECT_INTERCEPTED_FUNCTION(free_symbol, free, void, void *);
	REDIRECT_INTERCEPTED_FUNCTION(calloc_symbol, calloc, void *, size_t, size_t);
	REDIRECT_INTERCEPTED_FUNCTION(realloc_symbol, realloc, void *, void *, size_t);
	REDIRECT_INTERCEPTED_FUNCTION(valloc_symbol, valloc, void *, size_t);
	REDIRECT_INTERCEPTED_FUNCTION(memalign_symbol, memalign, void *, size_t, size_t);
	REDIRECT_INTERCEPTED_FUNCTION(pvalloc_symbol, pvalloc, void *, size_t);
	
}


void nanos6_stop_function_interception()
{
	malloc_symbol = __libc_malloc;
	free_symbol = __libc_free;
	calloc_symbol = __libc_calloc;
	realloc_symbol = __libc_realloc;
	valloc_symbol = __libc_valloc;
	memalign_symbol = __libc_memalign;
	pvalloc_symbol = __libc_pvalloc;
	
	posix_memalign_symbol = posix_memalign_libc_symbol;
	
#if HAVE_ALIGNED_ALLOC
	aligned_alloc_symbol = aligned_alloc_libc_symbol;
#endif
	
#if HAVE_REALLOCARRAY
	reallocarray_symbol = reallocarray_libc_symbol;
#endif
}


void nanos6_memory_allocation_interception_init()
{
	void (*init_function)() = (void (*)()) dlsym(_nanos6_lib_handle, "nanos6_memory_allocation_interception_init");
	
	if (init_function != NULL) {
		init_function();
	}
}


void nanos6_memory_allocation_interception_fini()
{
	void (*fini_function)() = (void (*)()) dlsym(_nanos6_lib_handle, "nanos6_memory_allocation_interception_fini");
	
	if (fini_function != NULL) {
		fini_function();
	}
}


#pragma GCC visibility pop

