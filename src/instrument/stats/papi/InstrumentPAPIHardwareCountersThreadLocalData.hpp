/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_PAPI_HARDWARE_COUNTERS_THREAD_LOCAL_DATA_HPP
#define INSTRUMENT_PAPI_HARDWARE_COUNTERS_THREAD_LOCAL_DATA_HPP


namespace InstrumentHardwareCounters {
	namespace PAPI {
		struct ThreadLocal;
	}
}


typedef InstrumentHardwareCounters::PAPI::ThreadLocal HardwareCountersThreadLocalData;


namespace InstrumentHardwareCounters {
	namespace PAPI {
		struct ThreadLocal;
		HardwareCountersThreadLocalData &getCurrentThreadHardwareCounters();
	}
}


#endif // INSTRUMENT_PAPI_HARDWARE_COUNTERS_THREAD_LOCAL_DATA_HPP
