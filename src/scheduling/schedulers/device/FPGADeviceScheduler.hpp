/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef FPGA_DEVICE_SCHEDULER_HPP
#define FPGA_DEVICE_SCHEDULER_HPP

#include "DeviceScheduler.hpp"
#include "SubDeviceScheduler.hpp"

class FPGADeviceScheduler : public DeviceScheduler {
	size_t _totalSubDevices;
	SubDeviceScheduler *_subDeviceSchedulers;
public:
	FPGADeviceScheduler(
		size_t totalComputePlaces,
		SchedulingPolicy policy,
		bool enablePriority,
		bool enableImmediateSuccessor,
		nanos6_device_t deviceType
	) :
		DeviceScheduler(totalComputePlaces, policy,
			enablePriority,	enableImmediateSuccessor,
			deviceType)
	{
		DeviceInfoImplementation *deviceInfo =
			static_cast<DeviceInfoImplementation*>(HardwareInfo::getDeviceInfo(_deviceType));
		_totalSubDevices = deviceInfo->getNumDevices();
		_subDeviceSchedulers =
			(SubDeviceScheduler *) MemoryAllocator::alloc(_totalSubDevices * sizeof(SubDeviceScheduler));
		for (size_t i = 0; i < _totalSubDevices; i++) {
			new (&_subDeviceSchedulers[deviceInfo->getDeviceSubType(i)])
				SubDeviceScheduler(totalComputePlaces, policy, enablePriority, enableImmediateSuccessor, _deviceType, deviceInfo->getDeviceSubType(i));
		}
	}

	virtual ~FPGADeviceScheduler()
	{
		for (size_t i = 0; i < _totalSubDevices; i++) {
			_subDeviceSchedulers[i].~SubDeviceScheduler();
		}
		MemoryAllocator::free(_subDeviceSchedulers, _totalSubDevices * sizeof(SubDeviceScheduler));
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

	Task *getReadyTask(ComputePlace *computePlace)
	{
		int subType = ((DeviceComputePlace *)computePlace)->getSubType();
		return _subDeviceSchedulers[subType].getReadyTask(computePlace);
	}

	inline std::string getName() const
	{
		return "FPGADeviceScheduler";
	}
};

#endif // FPGA_DEVICE_SCHEDULER_HPP
