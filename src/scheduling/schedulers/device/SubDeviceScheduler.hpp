/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef SUB_DEVICE_SCHEDULER_HPP
#define SUB_DEVICE_SCHEDULER_HPP

#include <string>

#include "DeviceScheduler.hpp"
#include "hardware/places/DeviceComputePlace.hpp"
#include "hardware/device/DeviceInfoImplementation.hpp"

class SubDeviceScheduler : public DeviceScheduler {
	size_t _totalSubDevices;
	int _deviceSubType;

public:
	SubDeviceScheduler(size_t totalComputePlaces, SchedulingPolicy policy, bool enablePriority, bool enableImmediateSuccessor, nanos6_device_t deviceType, int deviceSubType) :
		DeviceScheduler(totalComputePlaces, policy, enablePriority, enableImmediateSuccessor, deviceType),
		_deviceSubType(deviceSubType)
	{
		DeviceInfoImplementation *deviceInfo = static_cast<DeviceInfoImplementation*>(HardwareInfo::getDeviceInfo(deviceType));
		assert(deviceInfo != nullptr);

		Device *subDeviceType = deviceInfo->getDevice(_deviceSubType);
		assert(subDeviceType != nullptr);

		_totalSubDevices = subDeviceType->getNumDevices();
	}

	virtual ~SubDeviceScheduler()
	{}

	inline int getDeviceSubType()
	{
		return _deviceSubType;
	}

	Task *getReadyTask(ComputePlace *computePlace)
	{
		assert(computePlace != nullptr);
		assert(((DeviceComputePlace *)computePlace)->getSubType() == _deviceSubType);

		Task *result = getTask(computePlace);
		assert(result == nullptr || result->getDeviceSubType() == _deviceSubType);
		return result;
	}

	inline std::string getName() const
	{
		return "SubDeviceScheduler(" + std::to_string(_deviceSubType) + ")";
	}
};



#endif // SUB_DEVICE_SCHEDULER_HPP
