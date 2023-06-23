/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_ACCELERATOR_HPP
#define CUDA_ACCELERATOR_HPP

#include <list>

#include <nanos6/cuda_device.h>

#include "CUDAFunctions.hpp"
#include "CUDAStreamPool.hpp"
#include "hardware/device/Accelerator.hpp"
#include "support/config/ConfigVariable.hpp"
#include "tasks/Task.hpp"

class CUDAAccelerator : public Accelerator {
private:
	// Maximum number of kernel args before we allocate extra memory for them
	const static int MAX_STACK_ARGS = 16;

	// Name to not confuse with other more general events hadled in other portions of the runtime
	struct CUDAEvent {
		cudaEvent_t event;
		Task *task;
	};

	std::list<CUDAEvent> _activeEvents, _preallocatedEvents;
	cudaDeviceProp _deviceProperties;
	CUDAStreamPool _streamPool;

	// Whether the device service should run while there are running tasks
	static ConfigVariable<bool> _pinnedPolling;

	// The time period in microseconds between device service runs
	static ConfigVariable<size_t> _usPollingPeriod;

	// Whether the task data dependencies should be prefetched to the device
	static ConfigVariable<bool> _prefetchDataDependencies;

	// To be used in order to obtain the current task in nanos6_get_current_cuda_stream() call
	thread_local static Task *_currentTask;

	inline void generateDeviceEvironment(Task *task) override
	{
		// The Accelerator::runTask() function has already set the device so it's safe to proceed
		nanos6_cuda_device_environment_t &env = task->getDeviceEnvironment().cuda;
		env.stream = _streamPool.getCUDAStream();
		env.event = _streamPool.getCUDAEvent();
	}

	inline void finishTaskCleanup(Task *task) override
	{
		nanos6_cuda_device_environment_t &env = task->getDeviceEnvironment().cuda;
		_streamPool.releaseCUDAEvent(env.event);
		_streamPool.releaseCUDAStream(env.stream);
	}

	void acceleratorServiceLoop() override;

	void processCUDAEvents();

	void preRunTask(Task *task) override;

	void postRunTask(Task *task) override;

	void callTaskBody(Task *task, nanos6_address_translation_entry_t *translation);

public:
	CUDAAccelerator(int cudaDeviceIndex) :
		Accelerator(cudaDeviceIndex, nanos6_cuda_device),
		_streamPool(cudaDeviceIndex)
	{
		CUDAFunctions::getDeviceProperties(_deviceProperties, _deviceHandler);
	}

	~CUDAAccelerator()
	{
	}

	// Set current device as the active in the runtime
	inline void setActiveDevice() override
	{
		CUDAFunctions::setActiveDevice(_deviceHandler);
	}

	// In CUDA, the async FIFOs used are CUDA streams
	inline void *getAsyncHandle() override
	{
		return (void *)_streamPool.getCUDAStream();
	}

	inline void releaseAsyncHandle(void *stream) override
	{
		_streamPool.releaseCUDAStream((cudaStream_t)stream);
	}

	static inline Task *getCurrentTask()
	{
		return _currentTask;
	}
};

#endif // CUDA_ACCELERATOR_HPP
