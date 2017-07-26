/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef HARDWARE_COUNTERS_THREAD_LOCAL_DATA_HPP
#define HARDWARE_COUNTERS_THREAD_LOCAL_DATA_HPP


#if HAVE_PAPI
#include "PAPI/PAPIHardwareCountersThreadLocalData.hpp"
#else
#include "no-HC/NoHardwareCountersThreadLocalData.hpp"
#endif


#endif // HARDWARE_COUNTERS_THREAD_LOCAL_DATA_HPP
