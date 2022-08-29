/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_MEMORY_HPP
#define INSTRUMENT_MEMORY_HPP

namespace Instrument {

	// These are used to track when a new memory block is allocated or
	// deallocated.

	void memoryAllocEnter();
	void memoryAllocExit();
	void memoryFreeEnter();
	void memoryFreeExit();
}

#endif // INSTRUMENT_MEMORY_HPP
