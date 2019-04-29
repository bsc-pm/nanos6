/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef HOST_UNSYNC_SCHEDULER_HPP
#define HOST_UNSYNC_SCHEDULER_HPP

#include "scheduling/schedulers/UnsyncScheduler.hpp"

class Taskloop;

class HostUnsyncScheduler : public UnsyncScheduler {
	Taskloop *_currentTaskloop;
	
public:
	HostUnsyncScheduler(SchedulingPolicy policy, bool enablePriority, bool enableImmediateSuccessor)
		: UnsyncScheduler(policy, enablePriority, enableImmediateSuccessor), _currentTaskloop(nullptr)
	{}
	
	virtual ~HostUnsyncScheduler()
	{
		assert(_currentTaskloop == nullptr);
	}
	
	//! \brief Get a ready task for execution
	//!
	//! \param[in] computePlace the hardware place asking for scheduling orders
	//!
	//! \returns a ready task or nullptr
	Task *getReadyTask(ComputePlace *computePlace);
};


#endif // HOST_UNSYNC_SCHEDULER_HPP
