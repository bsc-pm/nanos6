/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_OVNI_THREAD_LOCAL_DATA_HPP
#define INSTRUMENT_OVNI_THREAD_LOCAL_DATA_HPP

#include <InstrumentInstrumentationContext.hpp>

namespace Instrument {
	// TODO can I remove this context?
	struct InstrumentationContext;

	struct ThreadLocalData {
		InstrumentationContext _context;
		bool hungry;
	};
}


#endif // INSTRUMENT_OVNI_THREAD_LOCAL_DATA_HPP
