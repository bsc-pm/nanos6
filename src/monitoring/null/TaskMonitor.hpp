/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef NULL_TASK_MONITOR_HPP
#define NULL_TASK_MONITOR_HPP


class TaskMonitor {

public:
	
	static inline void insertTimePerUnitOfCost(const std::string &, double)
	{
	}
	
	static inline void getAverageTimesPerUnitOfCost(
		std::vector<std::string> &,
		std::vector<double> &
	) {
	}
	
};

#endif // NULL_TASK_MONITOR_HPP
