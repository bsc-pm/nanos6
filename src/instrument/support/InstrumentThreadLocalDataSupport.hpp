#ifndef INSTRUMENT_THREAD_LOCAL_DATA_SUPPORT_HPP
#define INSTRUMENT_THREAD_LOCAL_DATA_SUPPORT_HPP


#include <InstrumentThreadLocalData.hpp>


namespace Instrument {
	struct ThreadLocalData;
	
	inline ThreadLocalData &getThreadLocalData();
}


#endif // INSTRUMENT_THREAD_LOCAL_DATA_SUPPORT_HPP
