/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_LOG_MESSAGE_HPP
#define INSTRUMENT_LOG_MESSAGE_HPP


#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>


namespace Instrument {
	template<typename... TS>
	void logMessage(InstrumentationContext const &context /* = ThreadInstrumentationContext::getCurrent() */, TS... components);
}


#endif // INSTRUMENT_LOG_MESSAGE_HPP
