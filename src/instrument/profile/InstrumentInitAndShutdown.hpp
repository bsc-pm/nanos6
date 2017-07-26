/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_PROFILE_INIT_AND_SHUTDOWN_HPP
#define INSTRUMENT_PROFILE_INIT_AND_SHUTDOWN_HPP


#include "../api/InstrumentInitAndShutdown.hpp"


namespace Instrument {
	void initialize();
	void shutdown();
}


#endif // INSTRUMENT_PROFILE_INIT_AND_SHUTDOWN_HPP
