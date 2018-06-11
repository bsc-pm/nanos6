/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_MEMORY_PLACE_HPP
#define CUDA_MEMORY_PLACE_HPP

#include "hardware/places/MemoryPlace.hpp"

#include <cuda_runtime_api.h>

class Task;

class CUDAMemoryPlace: public MemoryPlace {
public:
	CUDAMemoryPlace(int device, cudaDeviceProp &properties)
		: MemoryPlace(device, nanos6_device_t::nanos6_cuda_device) 
	{}
	
	~CUDAMemoryPlace()
	{}
	
	virtual void preRunTask(Task *task) = 0;
	virtual void runTask(Task *task) = 0;
	virtual void postRunTask(Task *task) = 0;
	
	static CUDAMemoryPlace *createCUDAMemory(std::string const &memoryMode, int device, cudaDeviceProp &properties);
};

#endif //CUDA_MEMORY_PLACE_HPP

