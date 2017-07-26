/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

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
