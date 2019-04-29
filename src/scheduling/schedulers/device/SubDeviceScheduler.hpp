/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/



#ifndef SUB_DEVICE_SCHEDULER_HPP
#define SUB_DEVICE_SCHEDULER_HPP

#include "DeviceScheduler.hpp"
#include "hardware/places/DeviceComputePlace.hpp"
#include "hardware/device/DeviceInfoImplementation.hpp"

class SubDeviceScheduler : public DeviceScheduler {
	size_t _totalSubDevices;
	int _deviceSubType;
	std::vector<ComputePlace *> _cpuToSubDevice;
	
public:
	SubDeviceScheduler(SchedulingPolicy policy, bool enablePriority, bool enableImmediateSuccessor, nanos6_device_t deviceType, int deviceSubType)
		: DeviceScheduler(policy, enablePriority, enableImmediateSuccessor, deviceType),
		_deviceSubType(deviceSubType), _cpuToSubDevice(_totalCPUs, nullptr)
	{
		DeviceInfoImplementation *deviceInfo = static_cast<DeviceInfoImplementation*>(HardwareInfo::getDeviceInfo(deviceType));
		assert(deviceInfo != nullptr);
		
		Device *subDeviceType = deviceInfo->getDevice(_deviceSubType);
		assert(subDeviceType != nullptr);
		
		_totalSubDevices = subDeviceType->getNumDevices();
	}
	
	virtual ~SubDeviceScheduler()
	{}
	
	ComputePlace *getCPUToDevice(uint64_t)
	{
		FatalErrorHandler::failIf(true, "This should never happen.");
		return nullptr;
	}
	
	void setCPUToDevice(uint64_t, ComputePlace *)
	{
		FatalErrorHandler::failIf(true, "This should never happen.");
	}
	
	ComputePlace *getCPUToSubDevice(uint64_t cpuIndex)
	{
		return _cpuToSubDevice[cpuIndex];
	}
	
	void setCPUToSubDevice(uint64_t cpuIndex, ComputePlace *deviceComputePlace)
	{
		assert(deviceComputePlace == nullptr || ((DeviceComputePlace *)deviceComputePlace)->getSubType() == _deviceSubType);
		_cpuToSubDevice[cpuIndex] = deviceComputePlace;
	}
	
	inline int getDeviceSubType()
	{
		return _deviceSubType;
	}
	
	Task *getReadyTask(ComputePlace *computePlace, ComputePlace *deviceComputePlace)
	{
		assert(deviceComputePlace != nullptr);
		assert(((DeviceComputePlace *)deviceComputePlace)->getSubType() == _deviceSubType);
		
		Task *result = getTask(computePlace, deviceComputePlace, false, true);
		assert(result == nullptr || result->getDeviceSubType() == _deviceSubType);
		return result;
	}
	
	inline std::string getName() const
	{
		return "SubDeviceScheduler" + _deviceSubType;
	}
};



#endif // SUB_DEVICE_SCHEDULER_HPP
