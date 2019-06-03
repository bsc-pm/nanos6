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
	
	if ((nanos6NextPrivateLoadingMemoryFreeBlock + size + sizeof(size_t)) > (nanos6PrivateLoadingMemoryBase + MAX_PRIVATE_LOADING_MEMORY)) {
		MEM_ALLOC_FAIL(size);
		return NULL;
	}
	
	size_t *sizePtr = (size_t *) nanos6NextPrivateLoadingMemoryFreeBlock;
	nanos6NextPrivateLoadingMemoryFreeBlock += sizeof(size_t);
	*sizePtr = size;
	
	void *result = nanos6NextPrivateLoadingMemoryFreeBlock;
	nanos6NextPrivateLoadingMemoryFreeBlock += size;
	
	return result;
}


static void nanos6_loader_free(__attribute__((unused)) void *ptr)
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
		char const *name = info->dlpi_name;
		if (name[0] == 0) {
			name = NULL;
		}
		
		void *handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
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
			fprintf(stderr, "Nanos6 loader: Warning: Could not load '%s' to look up symbol '%s': %s\n", info->dlpi_name, lookupInfo->name, dlerror());
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
		char const *name = info->dlpi_name;
		if (name[0] == 0) {
			name = NULL;
			fprintf(stderr, "\tMain program: ");
		} else {
			fprintf(stderr, "\t%s: ", info->dlpi_name);
		}
		
		void *handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
		if (handle != NULL) {
			void *current = dlsym(handle, lookupInfo->name);
			if (current != NULL) {
				Dl_info dlInfo;
				int rc = dladdr(current, &dlInfo);
				if (rc != 0) {
					fprintf(stderr, "[%p] -> %s", current, dlInfo.dli_fname);
				} else {
					fprintf(stderr, "[%p] -> ???", current);
				}
				
				if (lookupInfo->foundOurFunction) {
					fprintf(stderr, " after nanos6 version\n");
					lookupInfo->result = current;
				} else if (current == lookupInfo->ourFunction) {
					fprintf(stderr, " nanos6 version\n");
					lookupInfo->foundOurFunction = true;
				} else {
					fprintf(stderr, " before nanos6 version\n");
				}
			} else {
				fprintf(stderr, "does not contain the symbol\n");
			}
			dlclose(handle);
		} else {
			fprintf(stderr, "%s\n", dlerror());
		}
	}
	
	return 0;
}


static void *nanos6_loader_find_next_function(void *ourFunction, char const *name, bool silentFailure)
{
	next_function_lookup_info_t nextFunctionLookup = { name, ourFunction, NULL, false };
	nextFunctionLookup.result = dlsym(RTLD_NEXT, name);
	
	if (!silentFailure && (nextFunctionLookup.result == NULL)) {
		char const *error = dlerror();
		if (error == NULL) {
			fprintf(stderr, "Nanos6 loader: Error resolving '%s'.\n", name);
		} else {
			fprintf(stderr, "Nanos6 loader: Error resolving '%s': %s.\n", name, error);
		}
		fprintf(stderr, "\tThis happens if the library that provides that function is linked before Nanos6.\n");
		fprintf(stderr, "\tFor instance the C library.\n");
		fprintf(stderr, "Lookup trace follows:\n");
		nextFunctionLookup.foundOurFunction = false;
		dl_iterate_phdr(nanos6_find_next_function_error_tracer, (void *) &nextFunctionLookup);
		fprintf(stderr, "\n");
		handle_error();
		return NULL;
	}
	
	return nextFunctionLookup.result;
}


static void nanos6_loader_resolve_next_memory_allocation_functions();


static void *nanos6_loader_intercepted_malloc(size_t size)
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


static void nanos6_loader_intercepted_free(void *ptr)
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


static void *nanos6_loader_intercepted_calloc(size_t nmemb, size_t size)
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


static void *nanos6_loader_intercepted_realloc(void *ptr, size_t size)
{
	if (nanos6LoaderInMemoryInitialization) {
		size_t *size_ptr = ptr;
		size_ptr--;
		
		if (*size_ptr >= size) {
			return ptr;
		} else {
			void *result = malloc(size);
			if (result != NULL) {
				memcpy(result, ptr, *size_ptr);
			}
			return result;
		}
	} else if (nanos6MemoryFunctionsInitialized) {
		return nanos6MemoryFunctions.realloc(ptr, size);
	} else {
		nanos6_loader_resolve_next_memory_allocation_functions();
		return nextMemoryFunctions.realloc(ptr, size);
	}
}


static void *nanos6_loader_intercepted_valloc(size_t size)
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


static void *nanos6_loader_intercepted_memalign(size_t alignment, size_t size)
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


static void *nanos6_loader_intercepted_pvalloc(size_t size)
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


static int nanos6_loader_intercepted_posix_memalign(void **memptr, size_t alignment, size_t size)
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
static void *nanos6_loader_intercepted_aligned_alloc(size_t alignment, size_t size)
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
static void *nanos6_loader_intercepted_reallocarray(void *ptr, size_t nmemb, size_t size)
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


static void nanos6_loader_resolve_next_memory_allocation_functions()
{
	if (nextMemoryFunctionsInitialized) {
		return;
	}
	
	nanos6LoaderInMemoryInitialization = 1;
	
	nextMemoryFunctions.malloc = nanos6_loader_find_next_function(nanos6_loader_intercepted_malloc, "malloc", 0);
	nextMemoryFunctions.free = nanos6_loader_find_next_function(nanos6_loader_intercepted_free, "free", 0);
	nextMemoryFunctions.calloc = nanos6_loader_find_next_function(nanos6_loader_intercepted_calloc, "calloc", 0);
	nextMemoryFunctions.realloc = nanos6_loader_find_next_function(nanos6_loader_intercepted_realloc, "realloc", 0);
#if HAVE_REALLOCARRAY
	nextMemoryFunctions.reallocarray = nanos6_loader_find_next_function(nanos6_loader_intercepted_reallocarray, "reallocarray", 0);
#endif
	nextMemoryFunctions.posix_memalign = nanos6_loader_find_next_function(nanos6_loader_intercepted_posix_memalign, "posix_memalign", 0);
#if HAVE_ALIGNED_ALLOC
	nextMemoryFunctions.aligned_alloc = nanos6_loader_find_next_function(nanos6_loader_intercepted_aligned_alloc, "aligned_alloc", 0);
#endif
	nextMemoryFunctions.valloc = nanos6_loader_find_next_function(nanos6_loader_intercepted_valloc, "valloc", 0);
	nextMemoryFunctions.memalign = nanos6_loader_find_next_function(nanos6_loader_intercepted_memalign, "memalign", 0);
	nextMemoryFunctions.pvalloc = nanos6_loader_find_next_function(nanos6_loader_intercepted_pvalloc, "pvalloc", 0);
	
	nanos6LoaderInMemoryInitialization = 0;
	
	nextMemoryFunctionsInitialized = 1;
}


#pragma GCC visibility push(default)


void *malloc(size_t size) __attribute__((alias("nanos6_loader_intercepted_malloc")));
void free(void *ptr) __attribute__((alias("nanos6_loader_intercepted_free")));
void *calloc(size_t nmemb, size_t size) __attribute__((alias("nanos6_loader_intercepted_calloc")));
void *realloc(void *ptr, size_t size) __attribute__((alias("nanos6_loader_intercepted_realloc")));
void *valloc(size_t size) __attribute__((alias("nanos6_loader_intercepted_valloc")));
void *memalign(size_t alignment, size_t size) __attribute__((alias("nanos6_loader_intercepted_memalign")));
void *pvalloc(size_t size) __attribute__((alias("nanos6_loader_intercepted_pvalloc")));
int posix_memalign(void **memptr, size_t alignment, size_t size) __attribute__((alias("nanos6_loader_intercepted_posix_memalign")));
#if HAVE_ALIGNED_ALLOC
void *aligned_alloc(size_t alignment, size_t size) __attribute__((alias("nanos6_loader_intercepted_aligned_alloc")));
#endif
#if HAVE_REALLOCARRAY
void *reallocarray(void *ptr, size_t nmemb, size_t size) __attribute__((alias("nanos6_loader_intercepted_reallocarray")));
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


void nanos6_memory_allocation_interception_postinit(void)
{
	void (*postinit_function)() = (void (*)()) dlsym(_nanos6_lib_handle, "nanos6_memory_allocation_interception_postinit");
	if (postinit_function != NULL) {
		postinit_function();
	}
}


void nanos6_memory_allocation_interception_fini(void)
{
	void (*fini_function)() = (void (*)()) dlsym(_nanos6_lib_handle, "nanos6_memory_allocation_interception_fini");
	
	if (fini_function != NULL) {
		fini_function();
		nanos6MemoryFunctionsInitialized = 0;
	}
}


#pragma GCC visibility pop

