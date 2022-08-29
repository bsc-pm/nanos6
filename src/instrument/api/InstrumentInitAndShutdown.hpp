/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_INIT_AND_SHUTDOWN_HPP
#define INSTRUMENT_INIT_AND_SHUTDOWN_HPP

#include <cstdint>


namespace Instrument {
	void initialize();
	void shutdown();
	void preinitFinished();
	void addCPUs();

	int64_t getInstrumentStartTime();
}


#endif // INSTRUMENT_INIT_AND_SHUTDOWN_HPP
