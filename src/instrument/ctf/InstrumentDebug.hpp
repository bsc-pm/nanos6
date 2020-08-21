/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_DEBUG_HPP
#define INSTRUMENT_CTF_DEBUG_HPP

#include <cstdint>

#include "CTFTracepoints.hpp"
#include "instrument/api/InstrumentDebug.hpp"


namespace Instrument {

	inline void debugEnter(uint8_t id)
	{
		tp_debug_enter(id);
	}

	inline void debugExit()
	{
		tp_debug_exit();
	}

	inline void debugRegister(const char *name, uint8_t id)
	{
		tp_debug_register(name, id);
	}

}

#endif //INSTRUMENT_CTF_DEBUG_HPP
