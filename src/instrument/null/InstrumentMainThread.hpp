/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_MAINTHREAD_HPP
#define INSTRUMENT_NULL_MAINTHREAD_HPP

#include "instrument/api/InstrumentMainThread.hpp"

namespace Instrument {

	inline void mainThreadBegin() {}
	inline void mainThreadEnd() {}
}

#endif // INSTRUMENT_NULL_MAINTHREAD_HPP

