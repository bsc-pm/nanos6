/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <loader/malloc.h>
#include <lowlevel/SymbolResolver.hpp>

#include "InstrumentProfile.hpp"


void *nanos6_intercepted_malloc(size_t size)
{
	Instrument::Profile::lightweightDisableForCurrentThread();
	auto result = SymbolResolver<void *, size_t>::call("malloc", size);
	Instrument::Profile::lightweightEnableForCurrentThread();
	
	return result;
}

void nanos6_intercepted_free(void *ptr)
{
	if (ptr != nullptr) {
		Instrument::Profile::lightweightDisableForCurrentThread();
		SymbolResolver<void, void *>::call("free", ptr);
		Instrument::Profile::lightweightEnableForCurrentThread();
	}
}

void *nanos6_intercepted_calloc(size_t nmemb, size_t size)
{
	Instrument::Profile::lightweightDisableForCurrentThread();
	auto result = SymbolResolver<void *, size_t, size_t>::call("calloc", nmemb, size);
	Instrument::Profile::lightweightEnableForCurrentThread();
	
	return result;
}

void *nanos6_intercepted_realloc(void *ptr, size_t size)
{
	Instrument::Profile::lightweightDisableForCurrentThread();
	auto result = SymbolResolver<void *, void *, size_t>::call("realloc", ptr, size);
	Instrument::Profile::lightweightEnableForCurrentThread();
	
	return result;
}

void *nanos6_intercepted_reallocarray(void *ptr, size_t nmemb, size_t size)
{
	Instrument::Profile::lightweightDisableForCurrentThread();
	auto result = SymbolResolver<void *, void *, size_t, size_t>::call("reallocarray", ptr, nmemb, size);
	Instrument::Profile::lightweightEnableForCurrentThread();
	
	return result;
}

int nanos6_intercepted_posix_memalign(void **memptr, size_t alignment, size_t size)
{
	Instrument::Profile::lightweightDisableForCurrentThread();
	auto result = SymbolResolver<int, void **, size_t, size_t>::call("posix_memalign", memptr, alignment, size);
	Instrument::Profile::lightweightEnableForCurrentThread();
	
	return result;
}

void *nanos6_intercepted_aligned_alloc(size_t alignment, size_t size)
{
	Instrument::Profile::lightweightDisableForCurrentThread();
	auto result = SymbolResolver<void *, size_t, size_t>::call("alloc", alignment, size);
	Instrument::Profile::lightweightEnableForCurrentThread();
	
	return result;
}

void *nanos6_intercepted_valloc(size_t size)
{
	Instrument::Profile::lightweightDisableForCurrentThread();
	auto result = SymbolResolver<void *, size_t>::call("valloc", size);
	Instrument::Profile::lightweightEnableForCurrentThread();
	
	return result;
}

void *nanos6_intercepted_memalign(size_t alignment, size_t size)
{
	Instrument::Profile::lightweightDisableForCurrentThread();
	auto result = SymbolResolver<void *, size_t, size_t>::call("memalign", alignment, size);
	Instrument::Profile::lightweightEnableForCurrentThread();
	
	return result;
}

void *nanos6_intercepted_pvalloc(size_t size)
{
	Instrument::Profile::lightweightDisableForCurrentThread();
	auto result = SymbolResolver<void *, size_t>::call("pvalloc", size);
	Instrument::Profile::lightweightEnableForCurrentThread();
	
	return result;
}
