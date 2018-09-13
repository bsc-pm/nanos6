/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_BOOTSTRAP_H
#define NANOS6_BOOTSTRAP_H

#include <stddef.h>


#pragma GCC visibility push(default)

enum nanos6_bootstrap_api_t { nanos6_bootstrap_api = 2 };


#ifdef __cplusplus
extern "C" {
#endif


typedef struct {
	void *(*malloc)(size_t size);
	void (*free)(void *ptr);
	void *(*calloc)(size_t nmemb, size_t size);
	void *(*realloc)(void *ptr, size_t size);
	void *(*reallocarray)(void *ptr, size_t nmemb, size_t size);

	int (*posix_memalign)(void **memptr, size_t alignment, size_t size);
	void *(*aligned_alloc)(size_t alignment, size_t size);
	void *(*valloc)(size_t size);

	void *(*memalign)(size_t alignment, size_t size);
	void *(*pvalloc)(size_t size);
} nanos6_memory_allocation_functions_t;


//! \brief Initialize the runtime at least to the point that it will accept tasks
void nanos6_preinit(void);

//! \brief Continue with the rest of the runtime initialization
void nanos6_init(void);

//! \brief Force the runtime to be shut down
// 
// This function is used to shut down the runtime
void nanos6_shutdown(void);

//! \brief Initialize memory interception
//! 
//! This function initializes the memory allocation interception. The first parameter contains the original
//! memory allocation functions and can be used for chain-calling.
//! The second table must be filled out with the replacements, that will be enabled after this call.
//! 
//! \param[in] nextMemoryFunctions the non-intercepted table of functions
//! \param[out] nanos6MemoryFunctions the intercepted table of functions
void nanos6_memory_allocation_interception_init(
	nanos6_memory_allocation_functions_t const *nextMemoryFunctions,
	nanos6_memory_allocation_functions_t *nanos6MemoryFunctions
);

//! \brief Second phase of the memory interception initialization after it has been enabled
void nanos6_memory_allocation_interception_postinit();

// The following function is called so that the runtime can prepare for library unloading
void nanos6_memory_allocation_interception_fini();


#ifdef __cplusplus
}
#endif


#pragma GCC visibility pop

#endif /* NANOS6_BOOTSTRAP_H */
