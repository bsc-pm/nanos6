/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef OPENACC_ACCELERATOR_HPP
#define OPENACC_ACCELERATOR_HPP

#include <deque>

#include <nanos6/openacc_device.h>

#include "OpenAccFunctions.hpp"
#include "OpenAccQueuePool.hpp"
#include "hardware/device/Accelerator.hpp"

class OpenAccAccelerator : public Accelerator {
private:
	std::deque<OpenAccQueue *> _activeQueues;

	OpenAccQueuePool _queuePool;

	inline bool isQueueAvailable()
	{
		return _queuePool.isQueueAvailable();
	}

	// For OpenACC tasks, the environment actually contains just an int, which is the
	// *async* argument OpenACC expects. Mercurium reads the environment and converts
	// the found acc pragmas to e.g.:
	// from:	#pragma acc kernels
	// to:		#pragma acc kernels async(asyncId)
	inline void generateDeviceEvironment(Task *task) override
	{
		nanos6_openacc_device_environment_t &env = task->getDeviceEnvironment().openacc;
		OpenAccQueue *queue = _queuePool.getAsyncQueue();
		// Use the deviceData to pass the queue object to further stages without having to
		// iterate through all queues to detect the task that has it.
		task->setDeviceData((void *)queue);
		env.asyncId = queue->getQueueId();
	}

	inline void preRunTask(Task *task) override
	{
		OpenAccQueue *queue = (OpenAccQueue *)task->getDeviceData();
		assert(queue != nullptr);
		queue->setTask(task);
	}

	inline void postRunTask(Task *task) override
	{
		OpenAccQueue *queue = (OpenAccQueue *)task->getDeviceData();
		assert(queue != nullptr);
		_activeQueues.push_back(queue);
	}

	inline void finishTaskCleanup(Task *task) override
	{
		OpenAccQueue *queue = (OpenAccQueue *)task->getDeviceData();
		_queuePool.releaseAsyncQueue(queue);
	}

	void acceleratorServiceLoop() override;

	void processQueues();

public:
	OpenAccAccelerator(int openaccDeviceIndex) :
		Accelerator(openaccDeviceIndex, nanos6_openacc_device),
		_queuePool()
	{
	}

	~OpenAccAccelerator()
	{
	}

	// Set current device as the active in the runtime
	inline void setActiveDevice() override
	{
		OpenAccFunctions::setActiveDevice(_deviceHandler);
	}

	// In OpenACC, the async FIFOs used are asynchronous queues
	inline void *getAsyncHandle() override
	{
		return (void *)_queuePool.getAsyncQueue();
	}

	inline void releaseAsyncHandle(void *queue) override
	{
		_queuePool.releaseAsyncQueue((OpenAccQueue *)queue);
	}
};

#endif // OPENACC_ACCELERATOR_HPP

