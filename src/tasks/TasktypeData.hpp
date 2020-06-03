/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASKTYPE_DATA_HPP
#define TASKTYPE_DATA_HPP

#include "hardware-counters/TasktypeHardwareCounters.hpp"

//! \brief Use to hold data on a per-tasktype basis (i.e. Monitoring data,
//! instrumentation parameters, etc.)
class TasktypeData {

private:

	//! Statistics of hardware counters of this tasktype
	TasktypeHardwareCounters _hwCounters;

public:

	inline TasktypeData() :
		_hwCounters()
	{
	}

	inline TasktypeHardwareCounters &getHardwareCounters()
	{
		return _hwCounters;
	}
};

#endif // TASKTYPE_DATA_HPP
