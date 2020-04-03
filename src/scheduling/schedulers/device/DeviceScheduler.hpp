/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef DEVICE_SCHEDULER_HPP
#define DEVICE_SCHEDULER_HPP

#include "scheduling/schedulers/SyncScheduler.hpp"
#include "scheduling/schedulers/device/DeviceUnsyncScheduler.hpp"

class DeviceScheduler : public SyncScheduler {
public:
	DeviceScheduler(size_t totalComputePlaces, SchedulingPolicy policy, bool enablePriority, __attribute__((unused)) bool enableImmediateSuccessor, nanos6_device_t deviceType)
		: SyncScheduler(totalComputePlaces, deviceType)
	{
		// Immediate successor support for devices is not available yet.
		_scheduler = new DeviceUnsyncScheduler(policy, enablePriority, false);
	}

	virtual Task *getReadyTask(ComputePlace *computePlace) = 0;

	virtual std::string getName() const = 0;
};


#endif // DEVICE_SCHEDULER_HPP
