/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2024 Barcelona Supercomputing Center (BSC)
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
	std::vector<CUDADirectoryAgent *> _directoryAgents;

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

				CUDADirectoryAgent *DirectoryAgent = new CUDADirectoryAgent(accelerator);
				Directory::registerDevice(DirectoryAgent);
				_directoryAgents.push_back(DirectoryAgent);

				accelerator->setDirectoryAgent(DirectoryAgent);
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

	inline void initializeDeviceServices() override
	{
		for (CUDAAccelerator *accelerator : _accelerators) {
			assert(accelerator != nullptr);
			accelerator->initializeService();
		}
	}

	inline void shutdownDeviceServices() override
	{
		for (CUDAAccelerator *accelerator : _accelerators) {
			assert(accelerator != nullptr);
			accelerator->shutdownService();
		}
	}

	inline size_t getComputePlaceCount() const override
	{
		return _deviceCount;
	}

	inline ComputePlace *getComputePlace(int handler) const override
	{
		return _accelerators[handler]->getComputePlace();
	}

	inline size_t getMemoryPlaceCount() const override
	{
		return _deviceCount;
	}

	inline MemoryPlace *getMemoryPlace(int handler) const override
	{
		return _accelerators[handler]->getMemoryPlace();
	}

	inline DirectoryAgent *getDirectoryAgent(int handler) const override
	{
		return _directoryAgents[handler];
	}
};

#endif // CUDA_DEVICE_INFO_HPP
