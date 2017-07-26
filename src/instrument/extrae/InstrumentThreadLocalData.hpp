/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

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
