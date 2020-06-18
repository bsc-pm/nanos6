/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_HARDWARE_COUNTERS_INTERFACE_HPP
#define TASK_HARDWARE_COUNTERS_INTERFACE_HPP

#include "SupportedHardwareCounters.hpp"


class TaskHardwareCountersInterface {

public:

	virtual inline ~TaskHardwareCountersInterface()
	{
	}

	//! \brief Empty hardware counter structures
	virtual void clear() = 0;

	//! \brief Get the delta value of a HW counter
	//!
	//! \param[in] counterType The type of counter to get the delta from
	virtual uint64_t getDelta(HWCounters::counters_t counterType) = 0;

	//! \brief Get the accumulated value of a HW counter
	//!
	//! \param[in] counterType The type of counter to get the accumulation from
	virtual uint64_t getAccumulated(HWCounters::counters_t counterType) = 0;

};

#endif // TASK_HARDWARE_COUNTERS_INTERFACE_HPP
