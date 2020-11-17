/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_LEADER_THREAD_HPP
#define INSTRUMENT_CTF_LEADER_THREAD_HPP

#include <cassert>

#include "instrument/api/InstrumentLeaderThread.hpp"
#include "instrument/ctf/InstrumentThreadLocalData.hpp"
#include "ctfapi/CTFAPI.hpp"

namespace Instrument {

	inline void leaderThreadSpin()
	{
		CPULocalData *cpuLocalData = getCTFCPULocalData();
		CTFAPI::CTFStream *userStream = cpuLocalData->userStream;
		assert(userStream != nullptr);

		CTFAPI::flushCurrentVirtualCPUBufferIfNeeded(userStream, userStream);
	}

}


#endif // INSTRUMENT_CTF_LEADER_THREAD_HPP
