/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_EVENT
#define CUDA_EVENT

#include <cuda_runtime_api.h>
#include <iostream>

#include "lowlevel/cuda/CUDAErrorHandler.hpp"
#include "tasks/TaskDeviceData.hpp"
#include "tasks/Task.hpp"


class CUDAEvent {

private:
	Task *_task;
	cudaEvent_t _event;
	
public:
	CUDAEvent()
	{
		cudaError_t err = cudaEventCreateWithFlags(&_event, cudaEventDisableTiming);
		CUDAErrorHandler::handle(err, "When creating event");
	}
	
	~CUDAEvent()
	{
		cudaError_t err = cudaEventDestroy(_event);
		CUDAErrorHandler::handle(err, "When destroying event");
	}
	
	void setTask(Task *task)
	{
		_task = task;
	}
	Task *getTask()
	{
		return _task;
	}
	
	void record()
	{
		cudaError_t err = cudaEventRecord(_event, ((CUDADeviceData *) _task->getDeviceData())->_stream->getStream());
		CUDAErrorHandler::handle(err, "When recording event");
	}
	
	bool finished()
	{
		cudaError_t err = cudaEventQuery(_event);
		return CUDAErrorHandler::handleEvent(err, "When checking event status");
	}
	
};

#endif //CUDA_EVENT

