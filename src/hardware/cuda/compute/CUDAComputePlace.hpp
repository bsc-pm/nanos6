/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_COMPUTE_PLACE_HPP
#define CUDA_COMPUTE_PLACE_HPP

#include "hardware/places/ComputePlace.hpp"
#include "hardware/cuda/compute/stream/CUDAStreamPool.hpp"
#include "hardware/cuda/compute/synchronization/CUDAEventPool.hpp"

class Task;

class CUDAComputePlace: public ComputePlace {
private:
	using CUDAEventList = std::vector<CUDAEvent *>;
	
	CUDAStreamPool _streamPool;
	CUDAEventPool _eventPool;
	
	CUDAEventList _activeEvents;
	
	SpinLock _lock;
public:
	using CUDATaskList = std::vector<Task*>;
	
	CUDAComputePlace(int device);
	~CUDAComputePlace();
	
	//! \brief Returns a list of tasks which kernels have finished 
	CUDATaskList getListFinishedTasks();
	
	//! \brief Assign compute resources to a task 
	void preRunTask(Task *task);
	
	//! \brief Execute the body of a task 
	void runTask(Task *task);
		
	//! \brief Release the compute resources of the CUDA task 
	void postRunTask(Task *task);
};

#endif //CUDA_COMPUTE_PLACE_HPP

