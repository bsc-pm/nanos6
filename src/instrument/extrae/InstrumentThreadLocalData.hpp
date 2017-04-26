#ifndef INSTRUMENT_EXTRAE_THREAD_LOCAL_DATA_HPP
#define INSTRUMENT_EXTRAE_THREAD_LOCAL_DATA_HPP


#include <InstrumentInstrumentationContext.hpp>
#include "InstrumentThreadId.hpp"

#include <vector>


namespace Instrument {
	struct ThreadLocalData {
		thread_id_t *_currentThreadId;
		std::vector<int> _nestingLevels;
		
		InstrumentationContext _context;
		
		ThreadLocalData()
			: _currentThreadId(nullptr), _nestingLevels()
		{
		}
	};
}


#endif // INSTRUMENT_EXTRAE_THREAD_LOCAL_DATA_HPP
