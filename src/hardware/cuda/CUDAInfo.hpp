/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_INFO_HPP
#define CUDA_INFO_HPP

#include "CUDADevice.hpp"
#include "hardware/hwinfo/DeviceInfo.hpp"
#include "executors/cuda/CUDAPollingService.hpp"

#include <cuda_runtime_api.h>
#include <vector>


class CUDAInfo: public DeviceInfo {
private:
	std::vector<CUDADevice *> _devices;
	std::vector<CUDAPollingService *> _pollingServices;	
	
public:
	
	void initialize();	
	void shutdown();
	
	inline size_t getComputePlaceCount(void)
	{
		return _devices.size();
	}
	inline ComputePlace* getComputePlace(int index)
	{
		return _devices[index]->getComputePlace();
	}
	
	inline size_t getMemoryPlaceCount(void)
	{
		return _devices.size();
	}
	inline MemoryPlace* getMemoryPlace(int index)
	{
		return _devices[index]->getMemoryPlace();
	}
		
	inline CUDAPollingService *getPollingService(int index)
	{
		return _pollingServices[index];
	}
	
	static inline void setDevice(int device)
	{
		cudaError_t err = cudaSetDevice(device);
		CUDAErrorHandler::handle(err, "When setting CUDA device environment");
	}
};

#endif //CUDA_INFO_HPP
