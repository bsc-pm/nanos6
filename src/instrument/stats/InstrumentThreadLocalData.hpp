#ifndef INSTRUMENT_STATS_THREAD_LOCAL_DATA_HPP
#define INSTRUMENT_STATS_THREAD_LOCAL_DATA_HPP


#include <InstrumentInstrumentationContext.hpp>

#include "InstrumentStats.hpp"


namespace Instrument {
	struct ThreadLocalData {
		Stats::ThreadInfo _threadInfo;
		InstrumentationContext _context;
		
		ThreadLocalData()
			: _threadInfo(true)
		{
		}
	};
}


#endif // INSTRUMENT_STATS_THREAD_LOCAL_DATA_HPP
