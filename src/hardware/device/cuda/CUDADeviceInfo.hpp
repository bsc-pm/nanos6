/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_DEVICE_INFO_HPP
#define CUDA_DEVICE_INFO_HPP

#include "CUDAAccelerator.hpp"
#include "CUDAFunctions.hpp"

#include "hardware/hwinfo/DeviceInfo.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "hardware/places/MemoryPlace.hpp"

// This class provides the interface to be used by the runtime's Hardware Info;
// Not to be confused with the device properties (see CUDAFunctions class)

class CUDADeviceInfo : public DeviceInfo {
	std::vector<CUDAAccelerator *> _accelerators;

public:
	CUDADeviceInfo()
	{
		_deviceCount = 0;
		if (!CUDAFunctions::initialize())
			return;

		_deviceCount = CUDAFunctions::getDeviceCount();
		_accelerators.reserve(_deviceCount);

		if (_deviceCount > 0) {
			// Create an Accelerator instance for each physical device
			for (size_t i = 0; i < _deviceCount; ++i) {
				CUDAAccelerator *accelerator = new CUDAAccelerator(i);
				assert(accelerator != nullptr);
				_accelerators.push_back(accelerator);
			}

			_deviceInitialized = true;
		}
	}

	~CUDADeviceInfo()
	{
		for (CUDAAccelerator *accelerator : _accelerators) {
			assert(accelerator != nullptr);
			delete accelerator;
		}
	}

	inline size_t getComputePlaceCount() const
	{
		return _deviceCount;
	}

	ComputePlace *getComputePlace(int handler)
	{
		return _accelerators[handler]->getComputePlace();
	}

	inline size_t getMemoryPlaceCount() const
	{
		return _deviceCount;
	}

	MemoryPlace *getMemoryPlace(int handler)
	{
		return _accelerators[handler]->getMemoryPlace();
	}
};

#endif // CUDA_DEVICE_INFO_HPP
