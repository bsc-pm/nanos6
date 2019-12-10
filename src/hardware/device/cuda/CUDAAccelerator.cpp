/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/polling.h>

#include "CUDAAccelerator.hpp"
#include "hardware/HardwareInfo.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "hardware/places/MemoryPlace.hpp"
#include "scheduling/Scheduler.hpp"

#include <DataAccessRegistration.hpp>
#include <DataAccessRegistrationImplementation.hpp>

thread_local Task* CUDAAccelerator::_currentTask;

CUDAAccelerator::CUDAAccelerator(int cudaDeviceIndex) :
	Accelerator(cudaDeviceIndex, nanos6_cuda_device),
	_streamPool(cudaDeviceIndex)
{
	CUDAFunctions::getDeviceProperties(_deviceProperties, _deviceHandler);
	registerPolling();
}

int CUDAAccelerator::pollingService(void *data)
{
	CUDAAccelerator *accel = (CUDAAccelerator *)data;
	assert(accel != nullptr);

	accel->acceleratorServiceLoop();
	return 0;
}

// Now the private functions

void CUDAAccelerator::acceleratorServiceLoop()
{
	// Check if the thread running the service is a WorkerThread. nullptr means LeaderThread
	bool worker = (WorkerThread::getCurrentWorkerThread() != nullptr);

	Task *task = nullptr;

	// For CUDA we need the setDevice operation here, for the stream & event context-to-thread association
	setActiveDevice();
	do {
		if (_streamPool.streamAvailable()) {
			task = Scheduler::getReadyTask(_computePlace);
			if (task != nullptr) {
				runTask(task);
			}
		}
		processCUDAEvents();
	} while (!_activeEvents.empty() && worker);

	// If process was run by LeaderThread, request a WorkerThread to continue.
	if (!worker && (task != nullptr || !_activeEvents.empty())) {
		CPUManager::executeCPUManagerPolicy(nullptr, ADDED_TASKS, 1);
	}
}

// For each CUDA device task a CUDA stream is required for the asynchronous
// launch; To ensure kernel completion a CUDA event is 'recorded' on the stream
// right after the kernel is queued. Then when a cudaEventQuery call returns
// succefully, we can be sure that the kernel execution (and hence the task)
// has finished.

// Get a new CUDA event and queue it in the stream the task has launched
void CUDAAccelerator::postRunTask(Task *task)
{
	nanos6_cuda_device_environment_t &env =	task->getDeviceEnvironment().cuda;
	CUDAFunctions::recordEvent(env.event, env.stream);
	_activeEvents.push_back({env.event, task});
}

void CUDAAccelerator::preRunTask(Task *task)
{
	// set the thread_local static var to be used by nanos6_get_current_cuda_stream()
	CUDAAccelerator::_currentTask = task;

	// Prefetch available memory locations to the GPU
	nanos6_cuda_device_environment_t &env =	task->getDeviceEnvironment().cuda;

	DataAccessRegistration::processAllDataAccesses(task,
		[&](DataAccessRegion region, DataAccessType type,
			bool isWeak, __attribute__((unused)) MemoryPlace const *location) -> bool {
				if(type != REDUCTION_ACCESS_TYPE && !isWeak)
					CUDAFunctions::cudaDevicePrefetch(region.getStartAddress(), region.getSize(), _deviceHandler, env.stream, type == READ_ACCESS_TYPE);
				return true;
			});
}
// Query the events issued to detect task completion
void CUDAAccelerator::processCUDAEvents()
{
	_preallocatedEvents.clear();
	std::swap(_preallocatedEvents, _activeEvents);

	for (CUDAEvent &ev : _preallocatedEvents) {
		if (CUDAFunctions::cudaEventFinished(ev.event)) {
			finishTask(ev.task);
		} else {
			_activeEvents.push_back(ev);
		}
	}
}

