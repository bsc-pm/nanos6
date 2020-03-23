/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_DATA_ACCESSES_INFO_HPP
#define TASK_DATA_ACCESSES_INFO_HPP

#include <cstdlib>

class TaskDataAccessesInfo {
public:
	TaskDataAccessesInfo(__attribute__((unused)) size_t numDeps)
	{
	}

	inline size_t getAllocationSize()
	{
		return 0;
	}

	inline void setAllocationAddress(__attribute__((unused)) void *allocationAddress)
	{
	}
};

#endif // TASK_DATA_ACCESSES_INFO_HPP