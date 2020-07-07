/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "HardwareCounters.hpp"
#include "TaskHardwareCountersInfo.hpp"


TaskHardwareCounters::TaskHardwareCounters(const TaskHardwareCountersInfo &info) :
	_allocationAddress(info.getAllocationAddress()),
	_enabled(false)
{
}
