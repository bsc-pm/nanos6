/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif


#include "error.h"
#include "resolve.h"

#include <api/nanos6/bootstrap.h>

#include <dlfcn.h>
#include <errno.h>
#include <link.h>
#include <malloc.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

#ifndef HAVE_CONFIG_H
#include <config.h>
#endif


#define MAX_PRIVATE_LOADING_MEMORY  (4 * 1024 * 1024)

static int nanos6LoaderInMemoryInitialization = 0;
static char *nanos6PrivateLoadingMemoryBase = NULL;
static char *nanos6NextPrivateLoadingMemoryFreeBlock = NULL;


static int nextMemoryFunctionsInitialized = 0;
static int nanos6MemoryFunctionsInitialized = 0;
static nanos6_memory_allocation_functions_t nextMemoryFunctions = {
	NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
};
static nanos6_memory_allocation_functions_t nanos6MemoryFunctions = {
	NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
};


static char errorMessage[4096];

#define MEM_ALLOC_FAIL(size)  do { int bytes = snprintf(errorMessage, 4096, "Error: failed to allocate %lu bytes during nanos6 loader initialization (%s:%i)\n", size, __FILE__, __LINE__); write(1, errorMessage, bytes); } while (0)


static void *nanos6_loader_malloc(size_t size)
{
	if (nanos6PrivateLoadingMemoryBase == NULL) {
		nanos6PrivateLoadingMemoryBase = mmap(NULL, MAX_PRIVATE_LOADING_MEMORY, PROT_EXEC|PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
		
		if (nanos6PrivateLoadingMemoryBase == NULL) {
			MEM_ALLOC_FAIL(size);
			return NULL;
		}
		nanos6NextPrivateLoadingMemoryFreeBlock = nanos6PrivateLoadingMemoryBase;
	}
	
	if ((nanos6NextPrivateLoadingMemoryFreeBlock + size) > (nanos6PrivateLoadingMemoryBase + MAX_PRIVATE_LOADING_MEMORY)) {
		MEM_ALLOC_FAIL(size);
		return NULL;
	}
	
	void *result = nanos6NextPrivateLoadingMemoryFreeBlock;
	nanos6NextPrivateLoadingMemoryFreeBlock += size;
	
	return result;
}


static void *nanos6_loader_free(__attribute__((unused)) void *ptr)
{
}


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


static void nanos6_loader_resolve_next_memory_allocation_functions()
{
	if (nextMemoryFunctionsInitialized) {
		return;
	}
	
	nanos6LoaderInMemoryInitialization = 1;
	
	nextMemoryFunctions.malloc = nanos6_loader_find_next_function(malloc, "malloc", 0);
	nextMemoryFunctions.free = nanos6_loader_find_next_function(free, "free", 0);
	nextMemoryFunctions.calloc = nanos6_loader_find_next_function(calloc, "calloc", 0);
	nextMemoryFunctions.realloc = nanos6_loader_find_next_function(realloc, "realloc", 0);
#if HAVE_REALLOCARRAY
	nextMemoryFunctions.reallocarray = nanos6_loader_find_next_function(reallocarray, "reallocarray", 0);
#endif
	nextMemoryFunctions.posix_memalign = nanos6_loader_find_next_function(posix_memalign, "posix_memalign", 0);
#if HAVE_ALIGNED_ALLOC
	nextMemoryFunctions.aligned_alloc = nanos6_loader_find_next_function(aligned_alloc, "aligned_alloc", 0);
#endif
	nextMemoryFunctions.valloc = nanos6_loader_find_next_function(valloc, "valloc", 0);
	nextMemoryFunctions.memalign = nanos6_loader_find_next_function(memalign, "memalign", 0);
	nextMemoryFunctions.pvalloc = nanos6_loader_find_next_function(pvalloc, "pvalloc", 0);
	
	nanos6LoaderInMemoryInitialization = 0;
	
	nextMemoryFunctionsInitialized = 1;
}




#pragma GCC visibility push(default)


void *malloc(size_t size)
{
	if (nanos6LoaderInMemoryInitialization) {
		return nanos6_loader_malloc(size);
	} else if (nanos6MemoryFunctionsInitialized) {
		return nanos6MemoryFunctions.malloc(size);
	} else {
		nanos6_loader_resolve_next_memory_allocation_functions();
		return nextMemoryFunctions.malloc(size);
	}
}


void free(void *ptr)
{
	if (nanos6LoaderInMemoryInitialization) {
		nanos6_loader_free(ptr);
	} else if (
		(ptr >= (void *) nanos6PrivateLoadingMemoryBase)
		&& (ptr < (void *) nanos6NextPrivateLoadingMemoryFreeBlock)
	) {
		nanos6_loader_free(ptr);
	} else if (nanos6MemoryFunctionsInitialized) {
		nanos6MemoryFunctions.free(ptr);
	} else {
		nanos6_loader_resolve_next_memory_allocation_functions();
		nextMemoryFunctions.free(ptr);
	}
}


void *calloc(size_t nmemb, size_t size)
{
	if (nanos6LoaderInMemoryInitialization) {
		return nanos6_loader_malloc(nmemb*size);
	} else if (nanos6MemoryFunctionsInitialized) {
		return nanos6MemoryFunctions.calloc(nmemb, size);
	} else {
		nanos6_loader_resolve_next_memory_allocation_functions();
		return nextMemoryFunctions.calloc(nmemb, size);
	}
}


void *realloc(void *ptr, size_t size)
{
	if (nanos6LoaderInMemoryInitialization) {
		MEM_ALLOC_FAIL(size);
		return NULL;
	} else if (nanos6MemoryFunctionsInitialized) {
		return nanos6MemoryFunctions.realloc(ptr, size);
	} else {
		nanos6_loader_resolve_next_memory_allocation_functions();
		return nextMemoryFunctions.realloc(ptr, size);
	}
}


void *valloc(size_t size)
{
	if (nanos6LoaderInMemoryInitialization) {
		MEM_ALLOC_FAIL(size);
		return NULL;
	} else if (nanos6MemoryFunctionsInitialized) {
		return nanos6MemoryFunctions.valloc(size);
	} else {
		nanos6_loader_resolve_next_memory_allocation_functions();
		return nextMemoryFunctions.valloc(size);
	}
}


void *memalign(size_t alignment, size_t size)
{
	if (nanos6LoaderInMemoryInitialization) {
		MEM_ALLOC_FAIL(size);
		return NULL;
	} else if (nanos6MemoryFunctionsInitialized) {
		return nanos6MemoryFunctions.memalign(alignment, size);
	} else {
		nanos6_loader_resolve_next_memory_allocation_functions();
		return nextMemoryFunctions.memalign(alignment, size);
	}
}


void *pvalloc(size_t size)
{
	if (nanos6LoaderInMemoryInitialization) {
		MEM_ALLOC_FAIL(size);
		return NULL;
	} else if (nanos6MemoryFunctionsInitialized) {
		return nanos6MemoryFunctions.pvalloc(size);
	} else {
		nanos6_loader_resolve_next_memory_allocation_functions();
		return nextMemoryFunctions.pvalloc(size);
	}
}


int posix_memalign(void **memptr, size_t alignment, size_t size)
{
	if (nanos6LoaderInMemoryInitialization) {
		MEM_ALLOC_FAIL(size);
		*memptr = NULL;
		return ENOMEM;
	} else if (nanos6MemoryFunctionsInitialized) {
		return nanos6MemoryFunctions.posix_memalign(memptr, alignment, size);
	} else {
		nanos6_loader_resolve_next_memory_allocation_functions();
		return nextMemoryFunctions.posix_memalign(memptr, alignment, size);
	}
}


#if HAVE_ALIGNED_ALLOC
void *aligned_alloc(size_t alignment, size_t size)
{
	if (nanos6LoaderInMemoryInitialization) {
		MEM_ALLOC_FAIL(size);
		return NULL;
	} else if (nanos6MemoryFunctionsInitialized) {
		return nanos6MemoryFunctions.aligned_alloc(alignment, size);
	} else {
		nanos6_loader_resolve_next_memory_allocation_functions();
		return nextMemoryFunctions.aligned_alloc(alignment, size);
	}
}
#endif


#if HAVE_REALLOCARRAY
void *reallocarray(void *ptr, size_t nmemb, size_t size)
{
	if (nanos6LoaderInMemoryInitialization) {
		MEM_ALLOC_FAIL(size);
		return NULL;
	} else if (nanos6MemoryFunctionsInitialized) {
		return nanos6MemoryFunctions.reallocarray(ptr, nmemb, size);
	} else {
		nanos6_loader_resolve_next_memory_allocation_functions();
		return nextMemoryFunctions.reallocarray(ptr, nmemb, size);
	}
}
#endif


void nanos6_loader_memory_allocation_interception_init()
{
	nanos6_loader_resolve_next_memory_allocation_functions();
	nanos6_memory_allocation_interception_init(&nextMemoryFunctions, &nanos6MemoryFunctions);
}


void nanos6_memory_allocation_interception_init(nanos6_memory_allocation_functions_t const *nextMemoryFunctions, nanos6_memory_allocation_functions_t *nanos6MemoryFunctions)
{
	void (*init_function)(nanos6_memory_allocation_functions_t const *, nanos6_memory_allocation_functions_t *) =
		(void (*)(nanos6_memory_allocation_functions_t const *, nanos6_memory_allocation_functions_t *))
		dlsym(_nanos6_lib_handle, "nanos6_memory_allocation_interception_init");
	
	if (init_function != NULL) {
		init_function(nextMemoryFunctions, nanos6MemoryFunctions);
		nanos6MemoryFunctionsInitialized = 1;
	}
}


void nanos6_memory_allocation_interception_postinit()
{
	void (*postinit_function)() = (void (*)()) dlsym(_nanos6_lib_handle, "nanos6_memory_allocation_interception_postinit");
	if (postinit_function != NULL) {
		postinit_function();
	}
}


void nanos6_memory_allocation_interception_fini()
{
	void (*fini_function)() = (void (*)()) dlsym(_nanos6_lib_handle, "nanos6_memory_allocation_interception_fini");
	
	if (fini_function != NULL) {
		fini_function();
		nanos6MemoryFunctionsInitialized = 0;
	}
}


#pragma GCC visibility pop

