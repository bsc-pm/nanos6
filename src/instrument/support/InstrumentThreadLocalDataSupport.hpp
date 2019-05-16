/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_THREAD_LOCAL_DATA_SUPPORT_HPP
#define INSTRUMENT_THREAD_LOCAL_DATA_SUPPORT_HPP


#include <InstrumentExternalThreadLocalData.hpp>
#include <InstrumentThreadLocalData.hpp>


namespace Instrument {
	ExternalThreadLocalData &getExternalThreadLocalData();
	ThreadLocalData &getThreadLocalData();
	
	inline ThreadLocalData &getSentinelNonWorkerThreadLocalData()
	{
		static thread_local ThreadLocalData nonWorkerThreadLocalData;
		return nonWorkerThreadLocalData;
	}
}


#endif // INSTRUMENT_THREAD_LOCAL_DATA_SUPPORT_HPP
