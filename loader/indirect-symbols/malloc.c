#include "resolve.h"

#include <stddef.h>


// The following functions have strong __libc_ counterparts that we can use during initialization,
// since dlopen and dlsym also perform memory allocations
DECLARE_LIBC_FALLBACK(__libc_, malloc, void *, size_t);
DECLARE_LIBC_FALLBACK(__libc_, free, void, void *);
DECLARE_LIBC_FALLBACK(__libc_, calloc, void *, size_t, size_t);
DECLARE_LIBC_FALLBACK(__libc_, realloc, void *, void *, size_t);
DECLARE_LIBC_FALLBACK(__libc_, valloc, void *, size_t);
DECLARE_LIBC_FALLBACK(__libc_, memalign, void *, size_t, size_t);
DECLARE_LIBC_FALLBACK(__libc_, pvalloc, void *, size_t);


void *malloc(size_t size)
{
	DECLARE_INTERCEPTED_FUNCTION_POINTER(malloc_symbol, malloc, void *, size_t);
	RESOLVE_INTERCEPTED_FUNCTION_WITH_LIBC_FALLBACK(__libc_, malloc_symbol, malloc, void *, size_t);
	
	return (*malloc_symbol)(size);
}


void free(void *ptr)
{
	DECLARE_INTERCEPTED_FUNCTION_POINTER(free_symbol, free, void, void *);
	RESOLVE_INTERCEPTED_FUNCTION_WITH_LIBC_FALLBACK(__libc_, free_symbol, free, void, void *);
	
	(*free_symbol)(ptr);
}


void *calloc(size_t nmemb, size_t size)
{
	DECLARE_INTERCEPTED_FUNCTION_POINTER(calloc_symbol, calloc, void *, size_t, size_t);
	RESOLVE_INTERCEPTED_FUNCTION_WITH_LIBC_FALLBACK(__libc_, calloc_symbol, calloc, void *, size_t, size_t);
	
	return (*calloc_symbol)(nmemb, size);
}


void *realloc(void *ptr, size_t size)
{
	DECLARE_INTERCEPTED_FUNCTION_POINTER(realloc_symbol, realloc, void *, void *, size_t);
	RESOLVE_INTERCEPTED_FUNCTION_WITH_LIBC_FALLBACK(__libc_, realloc_symbol, realloc, void *, void *, size_t);
	
	return (*realloc_symbol)(ptr, size);
}


void *valloc(size_t size)
{
	DECLARE_INTERCEPTED_FUNCTION_POINTER(valloc_symbol, valloc, void *, size_t);
	RESOLVE_INTERCEPTED_FUNCTION_WITH_LIBC_FALLBACK(__libc_, valloc_symbol, valloc, void *, size_t);
	
	return (*valloc_symbol)(size);
}


void *memalign(size_t alignment, size_t size)
{
	DECLARE_INTERCEPTED_FUNCTION_POINTER(memalign_symbol, memalign, void *, size_t, size_t);
	RESOLVE_INTERCEPTED_FUNCTION_WITH_LIBC_FALLBACK(__libc_, memalign_symbol, memalign, void *, size_t, size_t);
	
	return (*memalign_symbol)(alignment, size);
}


void *pvalloc(size_t size)
{
	DECLARE_INTERCEPTED_FUNCTION_POINTER(pvalloc_symbol, pvalloc, void *, size_t);
	RESOLVE_INTERCEPTED_FUNCTION_WITH_LIBC_FALLBACK(__libc_, pvalloc_symbol, pvalloc, void *, size_t);
	
	return (*pvalloc_symbol)(size);
}


void *reallocarray(void *ptr, size_t nmemb, size_t size)
{
	typedef void *(*reallocarray_t)(void *, size_t, size_t);
	
	static reallocarray_t symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (reallocarray_t) _nanos6_resolve_intercepted_symbol_with_global_fallback("nanos6_intercepted_reallocarray", "reallocarray", "memory allocation");
	}
	
	return (symbol)(ptr, nmemb, size);
}


int posix_memalign(void **memptr, size_t alignment, size_t size)
{
	typedef int (*posix_memalign_t)(void **, size_t, size_t);
	
	static posix_memalign_t symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (posix_memalign_t) _nanos6_resolve_intercepted_symbol_with_global_fallback("nanos6_intercepted_posix_memalign", "posix_memalign", "memory allocation");
	}
	
	return (symbol)(memptr, alignment, size);
}


void *aligned_alloc(size_t alignment, size_t size)
{
	typedef void *(*aligned_alloc_t)(size_t, size_t);
	
	static aligned_alloc_t symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (aligned_alloc_t) _nanos6_resolve_intercepted_symbol_with_global_fallback("nanos6_intercepted_aligned_alloc", "aligned_alloc", "memory allocation");
	}
	
	return (symbol)(alignment, size);
}


