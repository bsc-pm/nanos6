#ifndef INSTRUMENT_STATS_INIT_AND_SHUTDOWN_HPP
#define INSTRUMENT_STATS_INIT_AND_SHUTDOWN_HPP


#include "../api/InstrumentInitAndShutdown.hpp"

#include "performance/HardwareCounters.hpp"


namespace Instrument {
	inline void initialize()
	{
		HardwareCounters::initialize();
	}
	
	void shutdown();
	
}


#endif // INSTRUMENT_STATS_INIT_AND_SHUTDOWN_HPP
