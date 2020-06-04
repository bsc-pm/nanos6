/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef OPENACC_DEVICE_INFO_HPP
#define OPENACC_DEVICE_INFO_HPP

#include "OpenAccAccelerator.hpp"
#include "OpenAccFunctions.hpp"

#include "hardware/hwinfo/DeviceInfo.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "hardware/places/MemoryPlace.hpp"

class OpenAccDeviceInfo : public DeviceInfo {
	std::vector<OpenAccAccelerator *> _accelerators;

public:
	OpenAccDeviceInfo()
	{
		_deviceCount = OpenAccFunctions::getDeviceCount();
		_accelerators.reserve(_deviceCount);

		if (_deviceCount > 0) {
			// Create an Accelerator instance for each physical device
			for (size_t i = 0; i < _deviceCount; ++i) {
				OpenAccAccelerator *accelerator = new OpenAccAccelerator(i);
				assert(accelerator != nullptr);
				_accelerators.push_back(accelerator);
			}

			_deviceInitialized = true;
		}
	}

	~OpenAccDeviceInfo()
	{
		for (OpenAccAccelerator *accelerator : _accelerators) {
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

#endif // OPENACC_DEVICE_INFO_HPP

