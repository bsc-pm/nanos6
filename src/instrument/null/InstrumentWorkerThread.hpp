/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_WORKERTHREAD_HPP
#define INSTRUMENT_NULL_WORKERTHREAD_HPP


#include "instrument/api/InstrumentWorkerThread.hpp"

namespace Instrument {

	inline void workerThreadSpins() {}
	inline void workerThreadObtainedTask() {}
	inline void workerProgressing() {}
	inline void workerResting() {}
	inline void workerThreadBusyWaits() {}
	inline void workerThreadBegin() {}
	inline void workerThreadEnd() {}
	inline void enterHandleTask() {}
	inline void exitHandleTask() {}
	inline void enterSwitchTo() {}
	inline void exitSwitchTo() {}
	inline void enterSuspend() {}
	inline void exitSuspend() {}
	inline void enterResume() {}
	inline void exitResume() {}
}

#endif // INSTRUMENT_NULL_WORKERTHREAD_HPP

