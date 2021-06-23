/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_DEBUG_HPP
#define INSTRUMENT_DEBUG_HPP

#include <cstdint>

namespace Instrument {

	void debugRegister(const char *name, uint8_t id);

	void debugEnter(uint8_t id);
	void debugTransition(uint8_t id);
	void debugExit();

}

#endif // INSTRUMENT_DEBUG_HPP
