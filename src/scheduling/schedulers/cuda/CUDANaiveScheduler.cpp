/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#include "CUDANaiveScheduler.hpp"

#include "tasks/Task.hpp"
#include "hardware/HardwareInfo.hpp"
#include "hardware/cuda/compute/CUDAComputePlace.hpp"

#include <algorithm>
#include <cassert>
#include <mutex>


CUDANaiveScheduler::CUDANaiveScheduler(__attribute__((unused)) int numaNodeIndex)
{
	// Populate idle queues
	for (unsigned int i = 0; i < HardwareInfo::getComputePlaceCount(nanos6_device_t::nanos6_cuda_device); ++i) {
		CUDAComputePlace *gpu = (CUDAComputePlace *) HardwareInfo::getComputePlace(nanos6_device_t::nanos6_cuda_device, i);
		_idleGpus.push_back(gpu);
	}
}


CUDANaiveScheduler::~CUDANaiveScheduler()
{
}


ComputePlace * CUDANaiveScheduler::addReadyTask(Task *task, __attribute__((unused)) ComputePlace *hardwarePlace, __attribute__((unused)) ReadyTaskHint hint, bool doGetIdle)
{	
	assert(task->getDeviceType() == nanos6_device_t::nanos6_cuda_device);
	
	std::lock_guard<SpinLock> guard(_globalLock);
	_readyTasks.push_front(task);
	
	return nullptr;
}


Task *CUDANaiveScheduler::getReadyTask(ComputePlace *computePlace, __attribute__((unused)) Task *currentTask, __attribute__((unused)) bool canMarkAsIdle, __attribute__((unused)) bool doWait)
{
	assert(computePlace->getType() == nanos6_device_t::nanos6_cuda_device);
	
	Task *task = nullptr;
	
	std::lock_guard<SpinLock> guard(_globalLock);
	
	// 1. Get a ready task
	if( !_readyTasks.empty() ){
		task = _readyTasks.front();
		_readyTasks.pop_front();
		
		assert(task != nullptr);
		
		return task;
	}
	
	return nullptr;	
}


ComputePlace *CUDANaiveScheduler::getIdleComputePlace(bool force)
{
	return nullptr;
}

std::string CUDANaiveScheduler::getName() const
{
	return "cuda-naive";
}

