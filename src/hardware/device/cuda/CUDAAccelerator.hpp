/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_ACCELERATOR_HPP
#define CUDA_ACCELERATOR_HPP

#include <list>

#include "CUDAFunctions.hpp"
#include "CUDAStreamPool.hpp"

#include <nanos6/cuda_device.h>

#include <tasks/Task.hpp>

#include <hardware/device/Accelerator.hpp>

class CUDAAccelerator : public Accelerator {
private:
	// Name to not confuse with other more general events hadled in other portions of the runtime
	struct CUDAEvent {
		cudaEvent_t event;
		Task *task;
	};

	std::list<CUDAEvent> _activeEvents, _preallocatedEvents;
	cudaDeviceProp _deviceProperties;
	CUDAStreamPool _streamPool;

	// To be used in order to obtain the current task in nanos6_get_current_cuda_stream() call
	thread_local static Task* _currentTask;

	inline void generateDeviceEvironment(Task *task) override
	{
		nanos6_cuda_device_environment_t &env =	task->getDeviceEnvironment().cuda;
		env.stream = _streamPool.getCUDAStream();
		env.event = _streamPool.getCUDAEvent();
	}

	inline void finishTaskCleanup(Task *task) override
	{
		nanos6_cuda_device_environment_t &env =	task->getDeviceEnvironment().cuda;
		_streamPool.releaseCUDAEvent(env.event);
		_streamPool.releaseCUDAStream(env.stream);
	}

	inline void registerPolling() override
	{
		nanos6_register_polling_service("CUDA polling service", pollingService, (void *)this);
	}

	inline void unregisterPolling() override
	{
		nanos6_unregister_polling_service("CUDA polling service", pollingService, (void *)this);
	}

	void acceleratorServiceLoop() override;

	void processCUDAEvents();

	void preRunTask(Task *task) override;

	void postRunTask(Task *task) override;

public:
	CUDAAccelerator(int cudaDeviceIndex);

	~CUDAAccelerator()
	{
		unregisterPolling();
	}

	static int pollingService(void *data);

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
