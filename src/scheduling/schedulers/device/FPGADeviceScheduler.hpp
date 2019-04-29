/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/



#ifndef FPGA_DEVICE_SCHEDULER_HPP
#define FPGA_DEVICE_SCHEDULER_HPP

#include "DeviceScheduler.hpp"
#include "SubDeviceScheduler.hpp"

class FPGADeviceScheduler : public DeviceScheduler {
	size_t _totalSubDevices;
	SubDeviceScheduler *_subDeviceSchedulers;
public:
	FPGADeviceScheduler(SchedulingPolicy policy, bool enablePriority, bool enableImmediateSuccessor, nanos6_device_t deviceType)
		: DeviceScheduler(policy, enablePriority, enableImmediateSuccessor, deviceType)
	{
		DeviceInfoImplementation *deviceInfo = static_cast<DeviceInfoImplementation*>(HardwareInfo::getDeviceInfo(_deviceType));
		_totalSubDevices = deviceInfo->getNumDevices();
		_subDeviceSchedulers = (SubDeviceScheduler *) MemoryAllocator::alloc(_totalSubDevices * sizeof(SubDeviceScheduler));
		for (size_t i = 0; i < _totalSubDevices; i++) {
			new (&_subDeviceSchedulers[deviceInfo->getDeviceSubType(i)])
				SubDeviceScheduler(policy, enablePriority, enableImmediateSuccessor, _deviceType, deviceInfo->getDeviceSubType(i));
		}
	}
	
	virtual ~FPGADeviceScheduler()
	{
		for (size_t i = 0; i < _totalSubDevices; i++) {
			_subDeviceSchedulers[i].~SubDeviceScheduler();
		}
		MemoryAllocator::free(_subDeviceSchedulers, _totalSubDevices * sizeof(SubDeviceScheduler));
	}
	
	ComputePlace *getCPUToDevice(uint64_t)
	{
		FatalErrorHandler::failIf(true, "This should never happen.");
		return nullptr;
	}
	
	void setCPUToDevice(uint64_t, ComputePlace *)
	{
		FatalErrorHandler::failIf(true, "This should never happen.");
	}
	
	inline int getDeviceSubType(int subType)
	{
		return _subDeviceSchedulers[subType].getDeviceSubType();
	}
	
	inline void addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint)
	{
		int subType = task->getDeviceSubType();
		_subDeviceSchedulers[subType].addReadyTask(task, computePlace, hint);
	}
	
	Task *getReadyTask(ComputePlace *computePlace, ComputePlace *deviceComputePlace)
	{
		int subType = ((DeviceComputePlace *)deviceComputePlace)->getSubType();
		return _subDeviceSchedulers[subType].getReadyTask(computePlace, deviceComputePlace);
	}
	
	inline std::string getName() const
	{
		return "FPGADeviceScheduler";
	}
};

#endif
