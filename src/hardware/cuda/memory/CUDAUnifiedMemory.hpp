/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_UNIFIED_MEMORY_HPP
#define CUDA_UNIFIED_MEMORY_HPP

#include "CUDAMemoryPlace.hpp"

class CUDAUnifiedMemory: public CUDAMemoryPlace {
public:
	CUDAUnifiedMemory(int device, cudaDeviceProp &properties):
		CUDAMemoryPlace(device, properties)	
	{}
	
	~CUDAUnifiedMemory()
	{}	
	
	virtual void preRunTask(Task *task);
	virtual void runTask(Task *task);
	virtual void postRunTask(Task *task);
};

#endif //CUDA_UNIFIED_MEMORY_HPP

