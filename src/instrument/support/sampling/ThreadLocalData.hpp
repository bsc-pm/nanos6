/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_SUPPORT_SAMPLING_THREAD_LOCAL_DATA_HPP
#define INSTRUMENT_SUPPORT_SAMPLING_THREAD_LOCAL_DATA_HPP


#include <time.h>


namespace Instrument {
namespace Sampling {


struct ThreadLocalData {
	int _lightweightDisableCount;
	int _disableCount;
	timer_t _profilingTimer;
	
	ThreadLocalData()
		: _lightweightDisableCount(0), _disableCount(1)
	{
	}
};


} // namespace Sampling
} // namespace Instrument


#endif // INSTRUMENT_SUPPORT_SAMPLING_THREAD_LOCAL_DATA_HPP
