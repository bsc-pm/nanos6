/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/



#ifndef DEVICE_SCHEDULER_HPP
#define DEVICE_SCHEDULER_HPP

#include "scheduling/schedulers/SyncScheduler.hpp"
#include "scheduling/schedulers/device/DeviceUnsyncScheduler.hpp"

class DeviceScheduler : public SyncScheduler {
protected:
	/* Members */
	nanos6_device_t _deviceType;
	
public:
	DeviceScheduler(SchedulingPolicy policy, bool enablePriority, __attribute__((unused)) bool enableImmediateSuccessor, nanos6_device_t deviceType)
		: _deviceType(deviceType)
	{
		// Immediate successor support for devices is not available yet.
		_scheduler = new DeviceUnsyncScheduler(policy, enablePriority, false);
	}
	
	virtual ~DeviceScheduler()
	{
		delete _scheduler;
	}
	
	virtual nanos6_device_t getDeviceType()
	{
		return _deviceType;
	}
	
	virtual ComputePlace *getCPUToDevice(uint64_t cpuIndex) = 0;
	
	virtual void setCPUToDevice(uint64_t cpuIndex, ComputePlace *deviceComputePlace) = 0;
	
	virtual Task *getReadyTask(ComputePlace *computePlace, ComputePlace *deviceComputePlace) = 0;
	
	virtual std::string getName() const = 0;
};


#endif // DEVICE_SCHEDULER_HPP
