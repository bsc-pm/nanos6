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
	SubDeviceScheduler(SchedulingPolicy policy, bool enablePriority, bool enableImmediateSuccessor, nanos6_device_t deviceType, int deviceSubType) :
		DeviceScheduler(policy, enablePriority, enableImmediateSuccessor, deviceType),
		_deviceSubType(deviceSubType),
		_cpuToSubDevice(_totalCPUs, nullptr)
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
	
	Task *getReadyTask(ComputePlace *computePlace, ComputePlace *deviceComputePlace)
	{
		assert(deviceComputePlace != nullptr);
		assert(((DeviceComputePlace *)deviceComputePlace)->getSubType() == _deviceSubType);
		
		Task *result = getTask(computePlace, deviceComputePlace);
		assert(result == nullptr || result->getDeviceSubType() == _deviceSubType);
		return result;
	}
	
	inline std::string getName() const
	{
		return "SubDeviceScheduler" + _deviceSubType;
	}
	
protected:
	inline ComputePlace *getRelatedComputePlace(uint64_t cpuIndex) const
	{
		return _cpuToSubDevice[cpuIndex];
	}
	
	inline void setRelatedComputePlace(uint64_t cpuIndex, ComputePlace *computePlace)
	{
		DeviceComputePlace *deviceComputePlace = (DeviceComputePlace *)computePlace;
		assert(deviceComputePlace == nullptr || deviceComputePlace->getSubType() == _deviceSubType);
		
		_cpuToSubDevice[cpuIndex] = deviceComputePlace;
	}
};



#endif // SUB_DEVICE_SCHEDULER_HPP
