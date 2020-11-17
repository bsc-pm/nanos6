/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_THREAD_LOCAL_DATA_HPP
#define INSTRUMENT_CTF_THREAD_LOCAL_DATA_HPP


#include "ctfapi/CTFTypes.hpp"

#include <InstrumentInstrumentationContext.hpp>

namespace Instrument {
	// TODO can I remove this context?
	struct InstrumentationContext;

	struct ThreadLocalData {
		InstrumentationContext _context;
		bool isBusyWaiting;
		ctf_timestamp_t schedulerLockTimestamp;
	};
}


#endif // INSTRUMENT_CTF_THREAD_LOCAL_DATA_HPP
