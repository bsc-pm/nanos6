/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_LEADER_THREAD_HPP
#define INSTRUMENT_CTF_LEADER_THREAD_HPP


#include "instrument/api/InstrumentLeaderThread.hpp"
#include "ctfapi/CTFAPI.hpp"

namespace Instrument {

	inline void leaderThreadSpin()
	{
		CTFAPI::flushCurrentVirtualCPUBufferIfNeeded();
	}

}


#endif // INSTRUMENT_CTF_LEADER_THREAD_HPP
