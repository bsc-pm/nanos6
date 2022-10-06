/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_OVNI_MEMORY_HPP
#define INSTRUMENT_OVNI_MEMORY_HPP

#include "instrument/api/InstrumentMemory.hpp"
#include "OvniTrace.hpp"

namespace Instrument {

	inline void memoryAllocEnter() { Ovni::memoryAllocEnter(); }
	inline void memoryAllocExit()  { Ovni::memoryAllocExit(); }
	inline void memoryFreeEnter()  { Ovni::memoryFreeEnter(); }
	inline void memoryFreeExit()   { Ovni::memoryFreeExit(); }
}

#endif // INSTRUMENT_OVNI_MEMORY_HPP

