/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_LOG_MESSAGE_HPP
#define INSTRUMENT_NULL_LOG_MESSAGE_HPP


#include "../api/InstrumentLogMessage.hpp"


namespace Instrument {
	template<typename... TS>
	inline void logMessage(
		__attribute__((unused)) InstrumentationContext const &context,
		__attribute__((unused)) TS... components
	) {
	}
}


#endif // INSTRUMENT_NULL_LOG_MESSAGE_HPP
