/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_INIT_AND_SHUTDOWN_HPP
#define INSTRUMENT_NULL_INIT_AND_SHUTDOWN_HPP

#include "instrument/api/InstrumentInitAndShutdown.hpp"


namespace Instrument {
	inline void initialize()
	{
	}

	inline void shutdown()
	{
	}

	inline void preinitFinished()
	{
	}

	inline int64_t getInstrumentStartTime()
	{
		return 0;
	}
}


#endif // INSTRUMENT_NULL_INIT_AND_SHUTDOWN_HPP
