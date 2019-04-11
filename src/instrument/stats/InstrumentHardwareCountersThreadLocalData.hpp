/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_HARDWARE_COUNTERS_THREAD_LOCAL_DATA_HPP
#define INSTRUMENT_HARDWARE_COUNTERS_THREAD_LOCAL_DATA_HPP


#if HAVE_PAPI
#include "papi/InstrumentPAPIHardwareCountersThreadLocalData.hpp"
#else
#include "null/InstrumentNullHardwareCountersThreadLocalData.hpp"
#endif


#endif // INSTRUMENT_HARDWARE_COUNTERS_THREAD_LOCAL_DATA_HPP
