/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_OVNI_THREAD_LOCAL_DATA_HPP
#define INSTRUMENT_OVNI_THREAD_LOCAL_DATA_HPP

#include <InstrumentInstrumentationContext.hpp>

namespace Instrument {
	enum worker_status_t {
		progressing,
		resting,
		absorbing
	};

	struct ThreadLocalData {
		InstrumentationContext _context;
		worker_status_t _workerStatus;

		ThreadLocalData() :
			_context(),
			_workerStatus(progressing)
		{
		}
	};
}


#endif // INSTRUMENT_OVNI_THREAD_LOCAL_DATA_HPP
