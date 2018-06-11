/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#include "CUDAHelper.hpp" 

#include <DataAccessRegistration.hpp>

#include "hardware/cuda/CUDAManager.hpp"
#include "scheduling/Scheduler.hpp"

#include "tasks/Task.hpp"
#include "tasks/TaskDeviceData.hpp"

#include "executors/threads/TaskFinalization.hpp"

#include <cuda_runtime_api.h>

CUDAHelper::CUDAHelper(CUDAComputePlace *computePlace, CUDAMemoryPlace *memoryPlace)
	: _computePlace(computePlace), _memoryPlace(memoryPlace)
{
	std::stringstream ss;
	ss << "CUDAHelper-" << _computePlace->getIndex();
	_serviceName = ss.str();
}	

CUDAHelper::~CUDAHelper()
{
}

void CUDAHelper::start()
{
	nanos_register_polling_service(_serviceName.c_str(), (nanos_polling_service_t) &CUDAHelper::runHelper, (void *) this);
}

void CUDAHelper::stop()
{
	nanos_unregister_polling_service(_serviceName.c_str(), (nanos_polling_service_t) &CUDAHelper::runHelper, (void *) this);
}

void CUDAHelper::finishTask(Task *task)
{
	_computePlace->postRunTask(task);
	_memoryPlace->postRunTask(task);
	
	CUDADeviceData *deviceData = (CUDADeviceData *) task->getDeviceData();
	delete deviceData;
	
	if (task->mustDelayDataAccessRelease()) {
		task->setDelayedDataAccessRelease(true);
		DataAccessRegistration::handleEnterTaskwait(task, _computePlace);
		if (!task->markAsFinished()) {
			task = nullptr;
			return;
		}
		
		DataAccessRegistration::handleExitTaskwait(task, _computePlace);
		task->increaseRemovalBlockingCount();
	}
	
	DataAccessRegistration::unregisterTaskDataAccesses(task, _computePlace);
	
	if (task->markAsFinishedAfterDataAccessRelease()) {
		TaskFinalization::disposeOrUnblockTask(task, _computePlace);
	}
}

void CUDAHelper::launchTask(Task *task)
{
	CUDAManager::setDevice(_computePlace->getIndex());
	
	CUDADeviceData *deviceData = new CUDADeviceData();
	task->setDeviceData((void *) deviceData);	
	
	_computePlace->preRunTask(task);
	_memoryPlace->preRunTask(task);
	
	_memoryPlace->runTask(task);
	_computePlace->runTask(task);
}

void CUDAHelper::run()
{
	// Discover finished kernels and free their dependencies 
	auto finishedTasks = _computePlace->getListFinishedTasks();
	
	auto it = finishedTasks.begin();
	while (it != finishedTasks.end()) {
		finishTask(*it);
		it = finishedTasks.erase(it);
	}
	assert(finishedTasks.empty());
	
	/* Check for ready tasks */
	Task *task = Scheduler::getReadyTask(_computePlace);
	while (task != nullptr) { 
		launchTask(task);
		task = Scheduler::getReadyTask(_computePlace);
	}
}

bool CUDAHelper::runHelper(void *helper_ptr)
{
	CUDAHelper *helper = (CUDAHelper *) helper_ptr;
	helper->run();
	return false;
}

