/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <loader/malloc.h>
#include <lowlevel/SymbolResolver.hpp>

#include "InstrumentProfile.hpp"


namespace Instrument {
	bool _profilingIsReady = false;
}


__attribute__((constructor(0)))
static void nanos6_memory_allocation_interception_init()
{
	SymbolResolver<void>::globalScopeCall("nanos6_start_function_interception");
}


__attribute__((destructor(0)))
static void nanos6_memory_allocation_interception_fini()
{
	SymbolResolver<void>::globalScopeCall("nanos6_stop_function_interception");
}


#pragma GCC visibility push(default)

void *nanos6_intercepted_malloc(size_t size)
{
	if (Instrument::_profilingIsReady) {
		Instrument::Profile::lightweightDisableForCurrentThread();
	}
	auto result = SymbolResolver<void *, size_t>::call("malloc", size);
	Instrument::Profile::lightweightEnableForCurrentThread();
	
	return result;
}

void nanos6_intercepted_free(void *ptr)
{
	if (ptr != nullptr) {
		if (Instrument::_profilingIsReady) {
			Instrument::Profile::lightweightDisableForCurrentThread();
		}
		SymbolResolver<void, void *>::call("free", ptr);
		Instrument::Profile::lightweightEnableForCurrentThread();
	}
}

void *nanos6_intercepted_calloc(size_t nmemb, size_t size)
{
	if (Instrument::_profilingIsReady) {
		Instrument::Profile::lightweightDisableForCurrentThread();
	}
	auto result = SymbolResolver<void *, size_t, size_t>::call("calloc", nmemb, size);
	if (Instrument::_profilingIsReady) {
		Instrument::Profile::lightweightEnableForCurrentThread();
	}
	
	return result;
}

void *nanos6_intercepted_realloc(void *ptr, size_t size)
{
	if (Instrument::_profilingIsReady) {
		Instrument::Profile::lightweightDisableForCurrentThread();
	}
	auto result = SymbolResolver<void *, void *, size_t>::call("realloc", ptr, size);
	if (Instrument::_profilingIsReady) {
		Instrument::Profile::lightweightEnableForCurrentThread();
	}
	
	return result;
}

void *nanos6_intercepted_reallocarray(void *ptr, size_t nmemb, size_t size)
{
	if (Instrument::_profilingIsReady) {
		Instrument::Profile::lightweightDisableForCurrentThread();
	}
	auto result = SymbolResolver<void *, void *, size_t, size_t>::call("reallocarray", ptr, nmemb, size);
	if (Instrument::_profilingIsReady) {
		Instrument::Profile::lightweightEnableForCurrentThread();
	}
	
	return result;
}

int nanos6_intercepted_posix_memalign(void **memptr, size_t alignment, size_t size)
{
	if (Instrument::_profilingIsReady) {
		Instrument::Profile::lightweightDisableForCurrentThread();
	}
	auto result = SymbolResolver<int, void **, size_t, size_t>::call("posix_memalign", memptr, alignment, size);
	if (Instrument::_profilingIsReady) {
		Instrument::Profile::lightweightEnableForCurrentThread();
	}
	
	return result;
}

void *nanos6_intercepted_aligned_alloc(size_t alignment, size_t size)
{
	if (Instrument::_profilingIsReady) {
		Instrument::Profile::lightweightDisableForCurrentThread();
	}
	auto result = SymbolResolver<void *, size_t, size_t>::call("aligned_alloc", alignment, size);
	if (Instrument::_profilingIsReady) {
		Instrument::Profile::lightweightEnableForCurrentThread();
	}
	
	return result;
}

void *nanos6_intercepted_valloc(size_t size)
{
	if (Instrument::_profilingIsReady) {
		Instrument::Profile::lightweightDisableForCurrentThread();
	}
	auto result = SymbolResolver<void *, size_t>::call("valloc", size);
	if (Instrument::_profilingIsReady) {
		Instrument::Profile::lightweightEnableForCurrentThread();
	}
	
	return result;
}

void *nanos6_intercepted_memalign(size_t alignment, size_t size)
{
	if (Instrument::_profilingIsReady) {
		Instrument::Profile::lightweightDisableForCurrentThread();
	}
	auto result = SymbolResolver<void *, size_t, size_t>::call("memalign", alignment, size);
	if (Instrument::_profilingIsReady) {
		Instrument::Profile::lightweightEnableForCurrentThread();
	}
	
	return result;
}

void *nanos6_intercepted_pvalloc(size_t size)
{
	if (Instrument::_profilingIsReady) {
		Instrument::Profile::lightweightDisableForCurrentThread();
	}
	auto result = SymbolResolver<void *, size_t>::call("pvalloc", size);
	if (Instrument::_profilingIsReady) {
		Instrument::Profile::lightweightEnableForCurrentThread();
	}
	
	return result;
}

#pragma GCC visibility pop
