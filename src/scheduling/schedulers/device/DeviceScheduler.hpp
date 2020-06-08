/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef DEVICE_SCHEDULER_HPP
#define DEVICE_SCHEDULER_HPP

#include "DeviceUnsyncScheduler.hpp"
#include "hardware/HardwareInfo.hpp"
#include "scheduling/schedulers/SyncScheduler.hpp"

class DeviceScheduler : public SyncScheduler {
private:
	// FIXME: currently unused
	size_t _totalDevices;

	std::string _name;

public:
	DeviceScheduler(
		size_t totalComputePlaces,
		SchedulingPolicy policy,
		bool enablePriority, bool,
		nanos6_device_t deviceType,
		std::string name
	) :
		SyncScheduler(totalComputePlaces, deviceType),
		_name(name)
	{
		_totalDevices = HardwareInfo::getComputePlaceCount(deviceType);

		// Immediate successor support for devices is not available yet
		_scheduler = new DeviceUnsyncScheduler(policy, enablePriority, false);
	}

	inline Task *getReadyTask(ComputePlace *computePlace)
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

private:
	inline ComputePlace *getComputePlace(uint64_t computePlaceIndex) const
	{
		return HardwareInfo::getComputePlace(_deviceType, computePlaceIndex);
	}

	inline bool mustStopServingTasks(ComputePlace *) const
	{
		return true;
	}

	inline void postServingTasks(ComputePlace *, Task *)
	{
	}
};


#endif // DEVICE_SCHEDULER_HPP
