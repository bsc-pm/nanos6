/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_OVNI_LEADER_THREAD_HPP
#define INSTRUMENT_OVNI_LEADER_THREAD_HPP

#include "instrument/api/InstrumentLeaderThread.hpp"
#include "OvniTrace.hpp"

namespace Instrument {
	inline void leaderThreadSpin()
	{
	}

	inline void leaderThreadBegin()
	{
		Ovni::threadTypeBegin('L');
	}

	inline void leaderThreadEnd()
	{
		Ovni::threadTypeEnd('L');
	}

}

#endif // INSTRUMENT_OVNI_LEADER_THREAD_HPP
