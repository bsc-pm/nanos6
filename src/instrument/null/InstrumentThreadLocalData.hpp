#ifndef INSTRUMENT_NULL_THREAD_LOCAL_DATA_HPP
#define INSTRUMENT_NULL_THREAD_LOCAL_DATA_HPP


#include <InstrumentInstrumentationContext.hpp>

namespace Instrument {
	struct InstrumentationContext;
	
	struct ThreadLocalData {
		InstrumentationContext _context;
	};
}


#endif // INSTRUMENT_NULL_THREAD_LOCAL_DATA_HPP
