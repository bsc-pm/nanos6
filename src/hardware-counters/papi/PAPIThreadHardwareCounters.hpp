/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef PAPI_THREAD_HARDWARE_COUNTERS_HPP
#define PAPI_THREAD_HARDWARE_COUNTERS_HPP

#include "hardware-counters/ThreadHardwareCountersInterface.hpp"


class PAPIThreadHardwareCounters : public ThreadHardwareCountersInterface {

public:

	inline PAPIThreadHardwareCounters()
	{
	}

};

#endif // PAPI_THREAD_HARDWARE_COUNTERS_HPP
