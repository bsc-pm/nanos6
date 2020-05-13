/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_THREAD_LOCAL_DATA_HPP
#define INSTRUMENT_CTF_THREAD_LOCAL_DATA_HPP


#include <InstrumentInstrumentationContext.hpp>

namespace Instrument {
	struct InstrumentationContext;

	struct ThreadLocalData {
		InstrumentationContext _context;
		bool isBusyWaiting;
	};
}


#endif // INSTRUMENT_CTF_THREAD_LOCAL_DATA_HPP
