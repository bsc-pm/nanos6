/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef NULL_TASK_HARDWARE_COUNTERS_MONITOR_HPP
#define NULL_TASK_HARDWARE_COUNTERS_MONITOR_HPP

#include "hardware-counters/SupportedHardwareCounters.hpp"


class TaskHardwareCountersMonitor {

public:
	
	static inline void insertCounterValuesPerUnitOfCost(
		const std::string &,
		std::vector<HWCounters::counters_t> &,
		std::vector<double> &
	) {
	}
	
	static inline void getAverageCounterValuesPerUnitOfCost(
		std::vector<std::string> &,
		std::vector<std::vector<std::pair<HWCounters::counters_t, double>>> &
	) {
	}
	
};

#endif // NULL_TASK_HARDWARE_COUNTERS_MONITOR_HPP
