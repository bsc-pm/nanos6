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

CUDAAccelerator::CUDAAccelerator(int cudaDeviceIndex) :
	Accelerator(cudaDeviceIndex, nanos6_cuda_device),
	_streamPool(cudaDeviceIndex)
{
	CUDAFunctions::getDeviceProperties(_deviceProperties, _deviceHandler);
	registerPolling();
}

// Now the private functions

bool CUDAAccelerator::acceleratorServiceLoop()
{
	if (_streamPool.streamAvailable()) {
		Task *task = Scheduler::getReadyTask(_computePlace);
		if (task != nullptr)
			runTask(task);
	}
	processCUDAEvents();
	return _active_events.size() != 0;
}

int CUDAAccelerator::polling(void *data)
{
	CUDAAccelerator *accel = (CUDAAccelerator *)data;
	assert(accel != nullptr);
	accel->setActiveDevice();

	while (accel->acceleratorServiceLoop());

	return 0;
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
	_active_events.push_back({env.event, task});
	// set the thread_local static var to be used by nanos6_get_current_cuda_stream()
	HardwareInfo::threadTask = task;
}

// Query the events issued to detect task completion
void CUDAAccelerator::processCUDAEvents()
{
	_preallocated_events.clear();
	std::swap(_preallocated_events, _active_events);

	for (CUDAEvent &ev : _preallocated_events)
		if (CUDAFunctions::cudaEventFinished(ev.event)) {
			finishTask(ev.task);
		}
		else
			_active_events.push_back(ev);
}

