/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef DEVICE_SCHEDULER_HPP
#define DEVICE_SCHEDULER_HPP

#include "scheduling/schedulers/SyncScheduler.hpp"
#include "scheduling/schedulers/device/DeviceUnsyncScheduler.hpp"

class DeviceScheduler : public SyncScheduler {
private:
	size_t _totalDevices;	// FIXME: currently unused
	std::string _name;

public:
	DeviceScheduler(
		size_t totalComputePlaces,
		SchedulingPolicy policy,
		bool enablePriority,
		__attribute__((unused)) bool enableImmediateSuccessor,
		nanos6_device_t deviceType,
		std::string name
	) :
		SyncScheduler(totalComputePlaces, deviceType),
		_name(name)
	{
		_totalDevices = HardwareInfo::getComputePlaceCount(deviceType); 	// FIXME: currently unused
		// Immediate successor support for devices is not available yet.
		_scheduler = new DeviceUnsyncScheduler(policy, enablePriority, false);
	}

	virtual Task *getReadyTask(ComputePlace *computePlace)
	{
		assert(computePlace != nullptr);
		assert(computePlace->getType() == _deviceType);

		Task *result = getTask(computePlace);
		assert(result == nullptr || result->getDeviceType() == _deviceType);
		return result;
	}

	inline std::string getName() const
	{
		return _name;
	}
};


#endif // DEVICE_SCHEDULER_HPP
