/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
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
	CUDADeviceScheduler(SchedulingPolicy policy, bool enablePriority, bool enableImmediateSuccessor, nanos6_device_t deviceType) :
		DeviceScheduler(policy, enablePriority, enableImmediateSuccessor, deviceType),
		_cpuToDevice(_totalCPUs, nullptr)
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
	
	ComputePlace *getCPUToDevice(uint64_t cpuIndex)
	{
		return _cpuToDevice[cpuIndex];
	}
	
	void setCPUToDevice(uint64_t cpuIndex, ComputePlace *deviceComputePlace)
	{
		assert(deviceComputePlace == nullptr || deviceComputePlace->getType() == _deviceType);
		_cpuToDevice[cpuIndex] = deviceComputePlace;
	}
	
	Task *getReadyTask(ComputePlace *computePlace, ComputePlace *deviceComputePlace)
	{
		assert(deviceComputePlace != nullptr);
		assert(deviceComputePlace->getType() == _deviceType);
		
		Task *result = getTask(computePlace, deviceComputePlace, true, false);
		assert(result == nullptr || result->getDeviceType() == _deviceType);
		return result;
	}
	
	inline std::string getName() const
	{
		return "CUDADeviceScheduler";
	}
};

#endif // CUDA_DEVICE_SCHEDULER_HPP
