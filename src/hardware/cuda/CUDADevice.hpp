/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_DEVICE_HPP
#define CUDA_DEVICE_HPP

#include "tasks/Task.hpp"

#include "compute/CUDAComputePlace.hpp"
#include "memory/CUDAMemoryPlace.hpp"

#include "hardware/places/NUMAPlace.hpp"

#include <cuda_runtime_api.h>
#include <vector>


class CUDADevice {
private:
	CUDAComputePlace * _computePlace; 		//!< ComputePlace associated to this device
	CUDAMemoryPlace * _memoryPlace; 		//!< MemoryPlace associated to this device
	
	int _index; 							//!< Index of the CUDA device in the cuda runtime
	NUMAPlace *_numaPlace;					//!< NUMA node where this device is located
public:

	CUDADevice(int index) :
		_index(index)
	{
		_computePlace = new CUDAComputePlace(index);
		_memoryPlace = new CUDAMemoryPlace(index);	
	}
	
	CUDAComputePlace *getComputePlace() const 
	{
		return _computePlace;
	}
	
	CUDAMemoryPlace *getMemoryPlace() const 
	{
		return _memoryPlace;
	}
	
	int getIndex() const 
	{
		return _index;
	}
	
	NUMAPlace *getNUMAPlace() const 
	{
		return _numaPlace;
	}
};

#endif //CUDA_DEVICE_HPP

