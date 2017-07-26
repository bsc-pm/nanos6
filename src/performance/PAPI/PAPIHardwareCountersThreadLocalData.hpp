/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef PAPI_HARDWARE_COUNTERS_THREAD_LOCAL_DATA_HPP
#define PAPI_HARDWARE_COUNTERS_THREAD_LOCAL_DATA_HPP


namespace HardwareCounters {
	namespace PAPI {
		struct ThreadLocal;
	}
}


typedef HardwareCounters::PAPI::ThreadLocal HardwareCountersThreadLocalData;


namespace HardwareCounters {
	namespace PAPI {
		struct ThreadLocal;
		inline HardwareCountersThreadLocalData &getCurrentThreadHardwareCounters();
	}
}


#endif // PAPI_HARDWARE_COUNTERS_THREAD_LOCAL_DATA_HPP
