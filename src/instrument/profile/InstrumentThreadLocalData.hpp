#ifndef INSTRUMENT_PROFILE_THREAD_LOCAL_DATA_HPP
#define INSTRUMENT_PROFILE_THREAD_LOCAL_DATA_HPP

#include <time.h>

#include <InstrumentInstrumentationContext.hpp>

#include "InstrumentProfile.hpp"


namespace Instrument {
	struct ThreadLocalData {
		int _disableCount;
		timer_t _profilingTimer;
		Profile::address_t *_currentBuffer;
		long _nextBufferPosition;
		
		ThreadLocalData()
			: _disableCount(1),
			_currentBuffer(nullptr), _nextBufferPosition(0)
		{
		}
	};
}


#endif // INSTRUMENT_PROFILE_THREAD_LOCAL_DATA_HPP
