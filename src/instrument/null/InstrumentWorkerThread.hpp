/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_WORKERTHREAD_HPP
#define INSTRUMENT_NULL_WORKERTHREAD_HPP


#include "../api/InstrumentWorkerThread.hpp"

namespace Instrument {

	inline void workerThreadSpins() {}
	inline void workerThreadObtainedTask() {}
	inline void workerThreadBusyWaits() {}
}

#endif // INSTRUMENT_NULL_WORKERTHREAD_HPP

