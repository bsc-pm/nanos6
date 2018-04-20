/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <api/nanos6/bootstrap.h>

#include "InstrumentExtrae.hpp"


namespace Instrument {
	bool _profilingIsReady = false;
}


static nanos6_memory_allocation_functions_t _nextMemoryFunctions;


#pragma GCC visibility push(default)


void *nanos6_intercepted_malloc(size_t size)
{
	if (Instrument::_profilingIsReady) {
		Instrument::Extrae::lightweightDisableSamplingForCurrentThread();
	}
	auto result = _nextMemoryFunctions.malloc(size);
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
		_nextMemoryFunctions.free(ptr);
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
	auto result = _nextMemoryFunctions.calloc(nmemb, size);
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
	auto result = _nextMemoryFunctions.realloc(ptr, size);
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
	auto result = _nextMemoryFunctions.reallocarray(ptr, nmemb, size);
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
	auto result = _nextMemoryFunctions.posix_memalign(memptr, alignment, size);
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
	auto result = _nextMemoryFunctions.aligned_alloc(alignment, size);
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
	auto result = _nextMemoryFunctions.valloc(size);
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
	auto result = _nextMemoryFunctions.memalign(alignment, size);
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
	auto result = _nextMemoryFunctions.pvalloc(size);
	if (Instrument::_profilingIsReady) {
		Instrument::Extrae::lightweightEnableSamplingForCurrentThread();
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


extern "C" void nanos6_memory_allocation_interception_fini()
{
}


#pragma GCC visibility pop


#include <instrument/support/InstrumentThreadLocalDataSupportImplementation.hpp>

