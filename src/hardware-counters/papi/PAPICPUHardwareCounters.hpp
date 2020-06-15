/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef PAPI_CPU_HARDWARE_COUNTERS_HPP
#define PAPI_CPU_HARDWARE_COUNTERS_HPP

#include "hardware-counters/CPUHardwareCountersInterface.hpp"
#include "hardware-counters/SupportedHardwareCounters.hpp"


class PAPICPUHardwareCounters : public CPUHardwareCountersInterface {

public:

	inline PAPICPUHardwareCounters()
	{
	}

	//! \brief Get the delta value of a HW counter
	//!
	//! \param[in] counterType The type of counter to get the delta from
	inline uint64_t getDelta(HWCounters::counters_t) override
	{
		return 0.0;
	}

};

#endif // PAPI_CPU_HARDWARE_COUNTERS_HPP
