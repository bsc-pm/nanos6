/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_LEADER_THREAD_HPP
#define INSTRUMENT_LEADER_THREAD_HPP

#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>


namespace Instrument {
	void leaderThreadSpin(InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
}


#endif // INSTRUMENT_LEADER_THREAD_HPP
