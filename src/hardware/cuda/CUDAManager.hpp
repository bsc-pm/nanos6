/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_MANAGER_HPP
#define CUDA_MANAGER_HPP

#include "tasks/Task.hpp"

#include "lowlevel/cuda/CUDAErrorHandler.hpp"
#include "compute/CUDAComputePlace.hpp"
#include "memory/CUDAMemoryPlace.hpp"
#include "executors/cuda/CUDAHelper.hpp"

#include <cuda_runtime_api.h>
#include <vector>


class CUDAManager {
private:
	static int _deviceCount;
	
	static std::vector<CUDAComputePlace *> _computePlaces;
	static std::vector<CUDAMemoryPlace *> _memoryPlaces;
	static std::vector<CUDAHelper *> _helpers;
	
	/* Private constructor */
	CUDAManager()
	{}	
	
public:
	
	static void initialize();	
	static void shutdown();
	
	static inline CUDAComputePlace *getComputePlace(int index)
	{
		return _computePlaces[index];
	}
	
	static inline CUDAMemoryPlace *getMemoryPlace(int index)
	{
		return _memoryPlaces[index];
	}
	
	static inline CUDAHelper *getHelper(int index)
	{
		return _helpers[index];
	}
	
	static inline int getDeviceCount()
	{
		return _deviceCount;
	}
	
	static inline void setDevice(int device)
	{
		cudaError_t err = cudaSetDevice(device);
		CUDAErrorHandler::handle(err, "When setting CUDA device environment");
	}
};

#endif //CUDA_MANAGER_HPP

