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
	CUDAMemoryPlace(int device)
		: MemoryPlace(device, nanos6_device_t::nanos6_cuda_device) 
	{}
	
	~CUDAMemoryPlace()
	{}
	
	void preRunTask(Task *task);
	void runTask(Task *task);
	void postRunTask(Task *task);
	
};

#endif //CUDA_MEMORY_PLACE_HPP

