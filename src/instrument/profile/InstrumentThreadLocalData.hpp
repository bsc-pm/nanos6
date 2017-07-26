/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_PROFILE_THREAD_LOCAL_DATA_HPP
#define INSTRUMENT_PROFILE_THREAD_LOCAL_DATA_HPP

#include <time.h>

#include <InstrumentInstrumentationContext.hpp>

#include "InstrumentProfile.hpp"


namespace Instrument {
	struct ThreadLocalData {
		int _lightweightDisableCount;
		int _disableCount;
		timer_t _profilingTimer;
		Profile::address_t *_currentBuffer;
		long _nextBufferPosition;
		
		ThreadLocalData()
			: _lightweightDisableCount(0), _disableCount(1),
			_currentBuffer(nullptr), _nextBufferPosition(0)
		{
		}
	};
}


#endif // INSTRUMENT_PROFILE_THREAD_LOCAL_DATA_HPP
