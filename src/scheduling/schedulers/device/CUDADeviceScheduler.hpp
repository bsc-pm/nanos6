/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_DEVICE_SCHEDULER_HPP
#define CUDA_DEVICE_SCHEDULER_HPP

#include "DeviceScheduler.hpp"
#include "hardware/places/DeviceComputePlace.hpp"
#include "hardware/device/DeviceInfoImplementation.hpp"

class CUDADeviceScheduler : public DeviceScheduler {
	size_t _totalDevices;
	std::vector<ComputePlace *> _cpuToDevice;
	
public:
	CUDADeviceScheduler(size_t totalComputePlaces, SchedulingPolicy policy, bool enablePriority, bool enableImmediateSuccessor, nanos6_device_t deviceType) :
		DeviceScheduler(totalComputePlaces, policy, enablePriority, enableImmediateSuccessor, deviceType),
		_cpuToDevice(_totalComputePlaces, nullptr)
	{
		DeviceInfoImplementation *deviceInfo = static_cast<DeviceInfoImplementation*>(HardwareInfo::getDeviceInfo(deviceType));
		assert(deviceInfo != nullptr);
		
		// CUDA has a single subtype.
		Device *subDeviceType = deviceInfo->getDevice(0);
		assert(subDeviceType != nullptr);
		
		_totalDevices = subDeviceType->getNumDevices();
	}
	
	virtual ~CUDADeviceScheduler()
	{}
	
	Task *getReadyTask(ComputePlace *computePlace)
	{
		assert(computePlace != nullptr);
		assert(computePlace->getType() == _deviceType);
		
		Task *result = getTask(computePlace);
		assert(result == nullptr || result->getDeviceType() == _deviceType);
		return result;
	}
	
	inline std::string getName() const
	{
		return "CUDADeviceScheduler";
	}
};

#endif // CUDA_DEVICE_SCHEDULER_HPP
