/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_PROFILE_THREAD_LOCAL_DATA_HPP
#define INSTRUMENT_PROFILE_THREAD_LOCAL_DATA_HPP

#include <InstrumentInstrumentationContext.hpp>
#include <instrument/support/sampling/ThreadLocalData.hpp>

#include "Address.hpp"


namespace Instrument {
	struct ThreadLocalData : public Instrument::Sampling::ThreadLocalData {
		address_t *_currentBuffer;
		long _nextBufferPosition;
		
		ThreadLocalData()
			: Instrument::Sampling::ThreadLocalData(),
			_currentBuffer(nullptr), _nextBufferPosition(0)
		{
		}
	};
}


#endif // INSTRUMENT_PROFILE_THREAD_LOCAL_DATA_HPP
