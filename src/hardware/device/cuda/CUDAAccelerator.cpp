/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "CUDAAccelerator.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "hardware/places/MemoryPlace.hpp"
#include "scheduling/Scheduler.hpp"
#include "system/BlockingAPI.hpp"

#include <DataAccessRegistration.hpp>
#include <DataAccessRegistrationImplementation.hpp>

thread_local Task* CUDAAccelerator::_currentTask;


void CUDAAccelerator::acceleratorServiceLoop()
{
	while (!shouldStopService()) {
		while (_streamPool.streamAvailable()) {
			Task *task = Scheduler::getReadyTask(_computePlace);
			if (task == nullptr)
				break;

			runTask(task);
		}

		// Only do the setActiveDevice if there have been tasks launched
		// Having setActiveDevice calls during e.g. bootstrap caused issues
		if (!_activeEvents.empty()) {
			setActiveDevice();
			processCUDAEvents();
		}

		// Sleep for 500 microseconds
		BlockingAPI::waitForUs(500);
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
		[&](const DataAccess *access) -> bool {
			if (access->getType() != REDUCTION_ACCESS_TYPE && !access->isWeak()) {
				CUDAFunctions::cudaDevicePrefetch(
					access->getAccessRegion().getStartAddress(),
					access->getAccessRegion().getSize(),
					_deviceHandler, env.stream,
					access->getType() == READ_ACCESS_TYPE);
			}
			return true;
		}
	);
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

