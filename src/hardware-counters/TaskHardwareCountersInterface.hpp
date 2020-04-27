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
	//! \param[in] counterId The type of counter to get the delta from
	virtual double getDelta(HWCounters::counters_t counterId) = 0;

	//! \brief Get the accumulated value of a HW counter
	//!
	//! \param[in] counterId The type of counter to get the accumulation from
	virtual double getAccumulated(HWCounters::counters_t counterId) = 0;

};

#endif // TASK_HARDWARE_COUNTERS_INTERFACE_HPP
