/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <api/nanos6/bootstrap.h>
#include <lowlevel/FatalErrorHandler.hpp>

#include "InstrumentProfile.hpp"


namespace Instrument {
	bool _profilingIsReady = false;
}


static void *_nonTlsAllocationCaller = nullptr;
static void *_tlsAllocationCaller = nullptr;
static thread_local int _tlsInitializationForcer;


static nanos6_memory_allocation_functions_t _nextMemoryFunctions;


#pragma GCC visibility push(default)

void *nanos6_intercepted_malloc(size_t size)
{
	// Tricks to detect the call to malloc to initialize the TLS
	if (!Instrument::_profilingIsReady) {
		if (_nonTlsAllocationCaller == nullptr) {
			_nonTlsAllocationCaller = __builtin_return_address(0);
		} else if (_tlsAllocationCaller == nullptr) {
			_tlsAllocationCaller = __builtin_return_address(0);
		}
	}
	
	bool mustDisableProfiling = (Instrument::_profilingIsReady && (__builtin_return_address(0) != _tlsAllocationCaller));
	if (mustDisableProfiling) {
		Instrument::Profile::lightweightDisableForCurrentThread();
	}
	auto result = _nextMemoryFunctions.malloc(size);
	if (mustDisableProfiling) {
		Instrument::Profile::lightweightEnableForCurrentThread();
	}
	
	return result;
}

void nanos6_intercepted_free(void *ptr)
{
	if (ptr != nullptr) {
		if (Instrument::_profilingIsReady) {
			Instrument::Profile::lightweightDisableForCurrentThread();
		}
		_nextMemoryFunctions.free(ptr);
		if (Instrument::_profilingIsReady) {
			Instrument::Profile::lightweightEnableForCurrentThread();
		}
	}
}

void *nanos6_intercepted_calloc(size_t nmemb, size_t size)
{
	if (Instrument::_profilingIsReady) {
		Instrument::Profile::lightweightDisableForCurrentThread();
	}
	auto result = _nextMemoryFunctions.calloc(nmemb, size);
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
	auto result = _nextMemoryFunctions.realloc(ptr, size);
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
	auto result = _nextMemoryFunctions.reallocarray(ptr, nmemb, size);
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
	auto result = _nextMemoryFunctions.posix_memalign(memptr, alignment, size);
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
	auto result = _nextMemoryFunctions.aligned_alloc(alignment, size);
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
	auto result = _nextMemoryFunctions.valloc(size);
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
	auto result = _nextMemoryFunctions.memalign(alignment, size);
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
	auto result = _nextMemoryFunctions.pvalloc(size);
	if (Instrument::_profilingIsReady) {
		Instrument::Profile::lightweightEnableForCurrentThread();
	}
	
	return result;
}


static const nanos6_memory_allocation_functions_t _nanos6MemoryFunctions = {
	.malloc = nanos6_intercepted_malloc,
	.free = nanos6_intercepted_free,
	.calloc = nanos6_intercepted_calloc,
	.realloc = nanos6_intercepted_realloc,
	.reallocarray = nanos6_intercepted_reallocarray,
	.posix_memalign = nanos6_intercepted_posix_memalign,
	.aligned_alloc = nanos6_intercepted_aligned_alloc,
	.valloc = nanos6_intercepted_valloc,
	.memalign = nanos6_intercepted_memalign,
	.pvalloc = nanos6_intercepted_pvalloc
};


extern "C" void nanos6_memory_allocation_interception_init(
	nanos6_memory_allocation_functions_t const *nextMemoryFunctions, 
	nanos6_memory_allocation_functions_t *nanos6MemoryFunctions
) {
	_nextMemoryFunctions = *nextMemoryFunctions;
	*nanos6MemoryFunctions = _nanos6MemoryFunctions;
}


extern "C" void nanos6_memory_allocation_interception_postinit(void)
{
	_nonTlsAllocationCaller = nullptr;
	_tlsAllocationCaller = nullptr;
	
	malloc(0);
	FatalErrorHandler::failIf(_nonTlsAllocationCaller == nullptr, "could not intercept malloc");
	FatalErrorHandler::failIf(_tlsAllocationCaller != nullptr, "spurious TLS memory allocation");
	
	_tlsInitializationForcer = 1;
	if (_tlsAllocationCaller == nullptr) {
		// In some configurations the TLS initialization calls to memory allocation functions in a non-interposable way
		_tlsAllocationCaller = (void *) ~ (size_t) 0;
	}
}


extern "C" void nanos6_memory_allocation_interception_fini(void)
{
}


#pragma GCC visibility pop

