/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef DEVICE_UNSYNC_SCHEDULER_HPP
#define DEVICE_UNSYNC_SCHEDULER_HPP

#include "scheduling/schedulers/UnsyncScheduler.hpp"

class DeviceUnsyncScheduler : public UnsyncScheduler {
public:
	DeviceUnsyncScheduler(SchedulingPolicy policy, bool enablePriority, bool enableImmediateSuccessor)
		: UnsyncScheduler(policy, enablePriority, enableImmediateSuccessor)
	{}
	
	virtual ~DeviceUnsyncScheduler()
	{}
	
	//! \brief Get a ready task for execution
	//!
	//! \param[in] computePlace the hardware place asking for scheduling orders
	//!
	//! \returns a ready task or nullptr
	Task *getReadyTask(ComputePlace *computePlace);
	
	//! \brief Check if the scheduler has available work for the current CPU
	//!
	//! \param[in] computePlace The host compute place
	bool hasAvailableWork(ComputePlace *computePlace);
};


#endif // DEVICE_UNSYNC_SCHEDULER_HPP
