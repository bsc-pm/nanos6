/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <loader/malloc.h>
#include <lowlevel/SymbolResolver.hpp>

#include "InstrumentExtrae.hpp"


namespace Instrument {
	bool _profilingIsReady = false;
}


static const StringLiteral _nanos6_start_function_interception_sl("nanos6_start_function_interception");
static const StringLiteral _nanos6_stop_function_interception_sl("nanos6_stop_function_interception");
static const StringLiteral _malloc_sl("malloc");
static const StringLiteral _free_sl("free");
static const StringLiteral _calloc_sl("calloc");
static const StringLiteral _realloc_sl("realloc");
static const StringLiteral _reallocarray_sl("reallocarray");
static const StringLiteral _posix_memalign_sl("posix_memalign");
static const StringLiteral _aligned_alloc_sl("aligned_alloc");
static const StringLiteral _valloc_sl("valloc");
static const StringLiteral _memalign_sl("memalign");
static const StringLiteral _pvalloc_sl("pvalloc");


#pragma GCC visibility push(default)

extern "C" void nanos6_memory_allocation_interception_init()
{
	// Resolve all the symbols before we intercept malloc, since the resolution itself does call malloc !?!
	SymbolResolver<void *, &_malloc_sl, size_t>::resolveNext();
	SymbolResolver<void, &_free_sl, void *>::resolveNext();
	SymbolResolver<void *, &_calloc_sl, size_t, size_t>::resolveNext();
	SymbolResolver<void *, &_realloc_sl, void *, size_t>::resolveNext();
	SymbolResolver<void *, &_reallocarray_sl, void *, size_t, size_t>::resolveNext();
	SymbolResolver<int, &_posix_memalign_sl, void **, size_t, size_t>::resolveNext();
	SymbolResolver<void *, &_aligned_alloc_sl, size_t, size_t>::resolveNext();
	SymbolResolver<void *, &_valloc_sl, size_t>::resolveNext();
	SymbolResolver<void *, &_memalign_sl, size_t, size_t>::resolveNext();
	SymbolResolver<void *, &_pvalloc_sl, size_t>::resolveNext();
	
	SymbolResolver<void, &_nanos6_start_function_interception_sl>::globalScopeCall();
}


extern "C" void nanos6_memory_allocation_interception_fini()
{
	SymbolResolver<void, &_nanos6_stop_function_interception_sl>::globalScopeCall();
}


void *nanos6_intercepted_malloc(size_t size)
{
	if (Instrument::_profilingIsReady) {
		Instrument::Extrae::lightweightDisableSamplingForCurrentThread();
	}
	auto result = SymbolResolver<void *, &_malloc_sl, size_t>::callNext(size);
	if (Instrument::_profilingIsReady) {
		Instrument::Extrae::lightweightEnableSamplingForCurrentThread();
	}
	
	return result;
}

void nanos6_intercepted_free(void *ptr)
{
	if (ptr != nullptr) {
		if (Instrument::_profilingIsReady) {
			Instrument::Extrae::lightweightDisableSamplingForCurrentThread();
		}
		SymbolResolver<void, &_free_sl, void *>::callNext(ptr);
		if (Instrument::_profilingIsReady) {
			Instrument::Extrae::lightweightEnableSamplingForCurrentThread();
		}
	}
}

void *nanos6_intercepted_calloc(size_t nmemb, size_t size)
{
	if (Instrument::_profilingIsReady) {
		Instrument::Extrae::lightweightDisableSamplingForCurrentThread();
	}
	auto result = SymbolResolver<void *, &_calloc_sl, size_t, size_t>::callNext(nmemb, size);
	if (Instrument::_profilingIsReady) {
		Instrument::Extrae::lightweightEnableSamplingForCurrentThread();
	}
	
	return result;
}

void *nanos6_intercepted_realloc(void *ptr, size_t size)
{
	if (Instrument::_profilingIsReady) {
		Instrument::Extrae::lightweightDisableSamplingForCurrentThread();
	}
	auto result = SymbolResolver<void *, &_realloc_sl, void *, size_t>::callNext(ptr, size);
	if (Instrument::_profilingIsReady) {
		Instrument::Extrae::lightweightEnableSamplingForCurrentThread();
	}
	
	return result;
}

void *nanos6_intercepted_reallocarray(void *ptr, size_t nmemb, size_t size)
{
	if (Instrument::_profilingIsReady) {
		Instrument::Extrae::lightweightDisableSamplingForCurrentThread();
	}
	auto result = SymbolResolver<void *, &_reallocarray_sl, void *, size_t, size_t>::callNext(ptr, nmemb, size);
	if (Instrument::_profilingIsReady) {
		Instrument::Extrae::lightweightEnableSamplingForCurrentThread();
	}
	
	return result;
}

int nanos6_intercepted_posix_memalign(void **memptr, size_t alignment, size_t size)
{
	if (Instrument::_profilingIsReady) {
		Instrument::Extrae::lightweightDisableSamplingForCurrentThread();
	}
	auto result = SymbolResolver<int, &_posix_memalign_sl, void **, size_t, size_t>::callNext(memptr, alignment, size);
	if (Instrument::_profilingIsReady) {
		Instrument::Extrae::lightweightEnableSamplingForCurrentThread();
	}
	
	return result;
}

void *nanos6_intercepted_aligned_alloc(size_t alignment, size_t size)
{
	if (Instrument::_profilingIsReady) {
		Instrument::Extrae::lightweightDisableSamplingForCurrentThread();
	}
	auto result = SymbolResolver<void *, &_aligned_alloc_sl, size_t, size_t>::callNext(alignment, size);
	if (Instrument::_profilingIsReady) {
		Instrument::Extrae::lightweightEnableSamplingForCurrentThread();
	}
	
	return result;
}

void *nanos6_intercepted_valloc(size_t size)
{
	if (Instrument::_profilingIsReady) {
		Instrument::Extrae::lightweightDisableSamplingForCurrentThread();
	}
	auto result = SymbolResolver<void *, &_valloc_sl, size_t>::callNext(size);
	if (Instrument::_profilingIsReady) {
		Instrument::Extrae::lightweightEnableSamplingForCurrentThread();
	}
	
	return result;
}

void *nanos6_intercepted_memalign(size_t alignment, size_t size)
{
	if (Instrument::_profilingIsReady) {
		Instrument::Extrae::lightweightDisableSamplingForCurrentThread();
	}
	auto result = SymbolResolver<void *, &_memalign_sl, size_t, size_t>::callNext(alignment, size);
	if (Instrument::_profilingIsReady) {
		Instrument::Extrae::lightweightEnableSamplingForCurrentThread();
	}
	
	return result;
}

void *nanos6_intercepted_pvalloc(size_t size)
{
	if (Instrument::_profilingIsReady) {
		Instrument::Extrae::lightweightDisableSamplingForCurrentThread();
	}
	auto result = SymbolResolver<void *, &_pvalloc_sl, size_t>::callNext(size);
	if (Instrument::_profilingIsReady) {
		Instrument::Extrae::lightweightEnableSamplingForCurrentThread();
	}
	
	return result;
}

#pragma GCC visibility pop


#include <instrument/support/InstrumentThreadLocalDataSupportImplementation.hpp>

