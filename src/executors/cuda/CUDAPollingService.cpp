/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#include "CUDAPollingService.hpp"

#include <DataAccessRegistration.hpp>

#include "hardware/cuda/CUDAInfo.hpp"
#include "scheduling/Scheduler.hpp"
#include "lowlevel/cuda/CUDAErrorHandler.hpp"

#include "tasks/Task.hpp"
#include "tasks/TaskDeviceData.hpp"
#include "tasks/TaskImplementation.hpp"

#include "executors/threads/TaskFinalization.hpp"

#include <cuda_runtime_api.h>

CUDAPollingService::CUDAPollingService(CUDADevice *device)
	: _device(device)
{
	std::stringstream ss;
	ss << "CUDAPollingService-" << _device->getIndex();
	_serviceName = ss.str();
}

CUDAPollingService::~CUDAPollingService()
{
}

void CUDAPollingService::start()
{
	nanos6_register_polling_service(_serviceName.c_str(), (nanos6_polling_service_t) &CUDAPollingService::runHelper, (void *) this);
}

void CUDAPollingService::stop()
{
	nanos6_unregister_polling_service(_serviceName.c_str(), (nanos6_polling_service_t) &CUDAPollingService::runHelper, (void *) this);
}

void CUDAPollingService::finishTask(Task *task)
{
	cudaError_t err = cudaPeekAtLastError();
	CUDAErrorHandler::handle(err);
	
	_device->getComputePlace()->postRunTask(task);
	_device->getMemoryPlace()->postRunTask(task);
	
	CUDADeviceData *deviceData = (CUDADeviceData *) task->getDeviceData();
	delete deviceData;
	
	if (task->markAsFinished(_device->getComputePlace())) {
		DataAccessRegistration::unregisterTaskDataAccesses(task, _device->getComputePlace(), _device->getComputePlace()->getDependencyData());
		
		if (task->markAsReleased()) {
			TaskFinalization::disposeOrUnblockTask(task, _device->getComputePlace());
		}
	}
}

void CUDAPollingService::launchTask(Task *task)
{
	assert(_device != nullptr);
	assert(task != nullptr);
	
	cudaSetDevice(_device->getIndex());
	
	CUDADeviceData *deviceData = new CUDADeviceData();
	task->setDeviceData((void *) deviceData);
	
	CUDAComputePlace *computePlace = _device->getComputePlace();
	assert(computePlace != nullptr);
	
	CUDAMemoryPlace *memoryPlace = _device->getMemoryPlace();
	assert(memoryPlace != nullptr);
	
	task->setMemoryPlace(memoryPlace);
	
	computePlace->preRunTask(task);
	memoryPlace->preRunTask(task);
	
	memoryPlace->runTask(task);
	computePlace->runTask(task);
}

void CUDAPollingService::run()
{
	// Discover finished kernels and free their dependencies
	auto finishedTasks = _device->getComputePlace()->getListFinishedTasks();
	
	auto it = finishedTasks.begin();
	while (it != finishedTasks.end()) {
		finishTask(*it);
		it = finishedTasks.erase(it);
	}
	assert(finishedTasks.empty());
	
	/* Check for ready tasks */
	Task *task = Scheduler::getReadyTask(_device->getComputePlace());
	while (task != nullptr) {
		launchTask(task);
		task = Scheduler::getReadyTask(_device->getComputePlace());
	}
}

bool CUDAPollingService::runHelper(void *service_ptr)
{
	CUDAPollingService *service = (CUDAPollingService *) service_ptr;
	service->run();
	return false;
}

