/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef PAPI_CPU_HARDWARE_COUNTERS_HPP
#define PAPI_CPU_HARDWARE_COUNTERS_HPP

#include "hardware-counters/CPUHardwareCountersInterface.hpp"


class PAPICPUHardwareCounters : public CPUHardwareCountersInterface {

public:

	inline PAPICPUHardwareCounters()
	{
	}

	inline double getDelta(HWCounters::counters_t)
	{
		return 0.0;
	}

};

#endif // PAPI_CPU_HARDWARE_COUNTERS_HPP
