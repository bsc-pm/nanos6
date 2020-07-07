/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_HARDWARE_COUNTERS_INFO_HPP
#define TASK_HARDWARE_COUNTERS_INFO_HPP

#include "TaskHardwareCounters.hpp"


class TaskHardwareCountersInfo {
private:
	//! The base allocation address used to store task hardware counters
	void *_allocationAddress;

	//! The size of task hardware counters
	size_t _allocationSize;

public:
	inline TaskHardwareCountersInfo() :
		_allocationAddress(nullptr),
		_allocationSize(TaskHardwareCounters::getTaskHardwareCountersSize())
	{
	}

	//! \brief Retreive the allocation address of task hardware counters
	inline void *getAllocationAddress() const
	{
		return _allocationAddress;
	}

	//! \brief Set the allocation address of task hardware counters
	//!
	//! \param[in] allocationAddress The new allocation address
	inline void setAllocationAddress(void *allocationAddress)
	{
		if (_allocationSize > 0) {
			_allocationAddress = allocationAddress;
		}
	}

	//! \brief Get the size needed to construct all the structures for all backends
	inline size_t getAllocationSize() const
	{
		return _allocationSize;
	}
};

#endif // TASK_HARDWARE_COUNTERS_INFO_HPP
