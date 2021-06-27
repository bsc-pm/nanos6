/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2021 Barcelona Supercomputing Center (BSC)
*/

#include "CUDAAccelerator.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "hardware/places/MemoryPlace.hpp"
#include "scheduling/Scheduler.hpp"
#include "system/BlockingAPI.hpp"

#include <DataAccessRegistration.hpp>
#include <DataAccessRegistrationImplementation.hpp>


ConfigVariable<bool> CUDAAccelerator::_pinnedPolling("devices.cuda.polling.pinned");
ConfigVariable<size_t> CUDAAccelerator::_usPollingPeriod("devices.cuda.polling.period_us");

thread_local Task* CUDAAccelerator::_currentTask;


void CUDAAccelerator::acceleratorServiceLoop()
{
	const size_t sleepTime = _usPollingPeriod.getValue();

	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);

	while (!shouldStopService()) {
		bool activeDevice = false;
		do {
			// Launch as many ready device tasks as possible
			while (_streamPool.streamAvailable()) {
				Task *task = Scheduler::getReadyTask(_computePlace, currentThread);
				if (task == nullptr)
					break;

				runTask(task);
			}

			// Only set the active device if there have been tasks launched
			// Setting the device during e.g. bootstrap caused issues
			if (!_activeEvents.empty()) {
				if (!activeDevice) {
					activeDevice = true;
					setActiveDevice();
				}

				// Process the active events
				processCUDAEvents();
			}

			// Iterate while there are running tasks and pinned polling is enabled
		} while (_pinnedPolling && !_activeEvents.empty());

		// Sleep for a configured amount of microseconds
		BlockingAPI::waitForUs(sleepTime);
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

