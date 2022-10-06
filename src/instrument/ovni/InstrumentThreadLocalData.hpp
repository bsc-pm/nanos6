/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_OVNI_THREAD_LOCAL_DATA_HPP
#define INSTRUMENT_OVNI_THREAD_LOCAL_DATA_HPP

#include <InstrumentInstrumentationContext.hpp>

namespace Instrument {
	struct ThreadLocalData {
		InstrumentationContext _context;
		bool _hungry;

		ThreadLocalData() :
			_context(),
			_hungry(false)
		{
		}
	};
}


#endif // INSTRUMENT_OVNI_THREAD_LOCAL_DATA_HPP
