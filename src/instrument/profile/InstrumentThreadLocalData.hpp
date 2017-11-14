/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_PROFILE_THREAD_LOCAL_DATA_HPP
#define INSTRUMENT_PROFILE_THREAD_LOCAL_DATA_HPP

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L
#endif


#include <InstrumentInstrumentationContext.hpp>
#include <instrument/support/sampling/ThreadLocalData.hpp>
#include <lowlevel/FatalErrorHandler.hpp>

#include "Address.hpp"

#include <cstdlib>


namespace Instrument {
	struct ThreadLocalData : public Instrument::Sampling::ThreadLocalData {
		address_t *_currentBuffer;
		long _nextBufferPosition;
		
		int _inMemoryAllocation;
		address_t *_nextBuffer;
		
		ThreadLocalData()
			: Instrument::Sampling::ThreadLocalData(),
			_currentBuffer(nullptr), _nextBufferPosition(0),
			_inMemoryAllocation(0), _nextBuffer(nullptr)
		{
		}
		
		void init(size_t bufferSize)
		{
			int rc = posix_memalign((void **) &_currentBuffer, 128, sizeof(address_t) * bufferSize);
			FatalErrorHandler::handle(rc, " allocating a buffer of ", sizeof(address_t) * bufferSize, " bytes for profiling");
		}
		
		void allocateNextBuffer(size_t bufferSize)
		{
			if (_nextBuffer == nullptr) {
				int rc = posix_memalign((void **) &_nextBuffer, 128, sizeof(address_t) * bufferSize);
				FatalErrorHandler::handle(rc, " allocating a buffer of ", sizeof(address_t) * bufferSize, " bytes for profiling");
			}
		}
	};
}


#endif // INSTRUMENT_PROFILE_THREAD_LOCAL_DATA_HPP
