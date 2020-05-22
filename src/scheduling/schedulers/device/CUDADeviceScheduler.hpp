/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_DEVICE_SCHEDULER_HPP
#define CUDA_DEVICE_SCHEDULER_HPP

#include "DeviceScheduler.hpp"

class CUDADeviceScheduler : public DeviceScheduler {
	size_t _totalDevices;

public:
	CUDADeviceScheduler(
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
		_totalDevices = HardwareInfo::getComputePlaceCount(deviceType);
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
