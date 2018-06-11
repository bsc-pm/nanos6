#include "CUDANaiveScheduler.hpp"

#include "tasks/Task.hpp"
#include "hardware/cuda/compute/CUDAComputePlace.hpp"
#include "hardware/cuda/CUDAManager.hpp"

#include <algorithm>
#include <cassert>
#include <mutex>


CUDANaiveScheduler::CUDANaiveScheduler(__attribute__((unused)) int numaNodeIndex)
{
	// Populate idle queues
	for(unsigned int i = 0; i < CUDAManager::getDeviceCount(); ++i){
		CUDAComputePlace *gpu = CUDAManager::getComputePlace(i);
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


void CUDANaiveScheduler::taskGetsUnblocked(Task *unblockedTask, __attribute__((unused)) ComputePlace *hardwarePlace)
{
}


Task *CUDANaiveScheduler::getReadyTask(ComputePlace *computePlace, __attribute__((unused)) Task *currentTask, bool canMarkAsIdle)
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

