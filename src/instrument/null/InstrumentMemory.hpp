/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_MEMORY_HPP
#define INSTRUMENT_NULL_MEMORY_HPP

#include "instrument/api/InstrumentMemory.hpp"

namespace Instrument {

	inline void memoryAllocEnter() { }
	inline void memoryAllocExit() { }
	inline void memoryFreeEnter() { }
	inline void memoryFreeExit() { }
}

#endif // INSTRUMENT_NULL_MEMORY_HPP

