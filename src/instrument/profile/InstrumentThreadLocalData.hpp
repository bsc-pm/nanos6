#ifndef INSTRUMENT_PROFILE_THREAD_LOCAL_DATA_HPP
#define INSTRUMENT_PROFILE_THREAD_LOCAL_DATA_HPP


#include <InstrumentInstrumentationContext.hpp>


namespace Instrument {
	struct ThreadLocalData {
		bool _enabled;
		
		ThreadLocalData()
			: _enabled(true)
		{
		}
	};
}


#endif // INSTRUMENT_PROFILE_THREAD_LOCAL_DATA_HPP
