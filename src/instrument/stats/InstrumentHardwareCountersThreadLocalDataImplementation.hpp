/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_HARDWARE_COUNTERS_THREAD_LOCAL_DATA_IMPLEMENTATION_HPP
#define INSTRUMENT_HARDWARE_COUNTERS_THREAD_LOCAL_DATA_IMPLEMENTATION_HPP


#if HAVE_PAPI
#include "papi/InstrumentPAPIHardwareCountersThreadLocalDataImplementation.hpp"
#else
#include "null/InstrumentNullHardwareCountersThreadLocalDataImplementation.hpp"
#endif


#endif // INSTRUMENT_HARDWARE_COUNTERS_THREAD_LOCAL_DATA_IMPLEMENTATION_HPP
