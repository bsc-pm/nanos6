/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_INIT_AND_SHUTDOWN_HPP
#define INSTRUMENT_CTF_INIT_AND_SHUTDOWN_HPP

#include "instrument/api/InstrumentInitAndShutdown.hpp"


namespace Instrument {
	void initialize();
	void shutdown();
	void preinitFinished();
	void addCPUs();
}


#endif // INSTRUMENT_CTF_INIT_AND_SHUTDOWN_HPP
