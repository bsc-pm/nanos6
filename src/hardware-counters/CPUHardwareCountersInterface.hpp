/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_HARDWARE_COUNTERS_INTERFACE_HPP
#define CPU_HARDWARE_COUNTERS_INTERFACE_HPP

#include "SupportedHardwareCounters.hpp"


class CPUHardwareCountersInterface {

public:

	virtual inline ~CPUHardwareCountersInterface()
	{
	}

	//! \brief Get the delta value of a HW counter
	//!
	//! \param[in] counterType The type of counter to get the delta from
	virtual uint64_t getDelta(HWCounters::counters_t counterType) const = 0;

};

#endif // CPU_HARDWARE_COUNTERS_INTERFACE_HPP
