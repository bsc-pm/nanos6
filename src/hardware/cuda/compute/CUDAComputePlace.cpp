/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#include "CUDAComputePlace.hpp"
#include "../memory/CUDAMemoryPlace.hpp"

#include "tasks/Task.hpp"
#include "tasks/TaskDeviceData.hpp"

#include <nanos6/cuda_device.h>

#ifndef NDEBUG
#include <cuda_runtime_api.h>
#endif

CUDAComputePlace::CUDAComputePlace(int device)
	: ComputePlace(device, nanos6_device_t::nanos6_cuda_device)
{
}

CUDAComputePlace::~CUDAComputePlace()
{
}

CUDAComputePlace::CUDATaskList CUDAComputePlace::getListFinishedTasks()
{
	std::lock_guard<SpinLock> guard(_lock);
	
	CUDATaskList finishedTasks;
	
	auto it = _activeEvents.begin();
	while (it != _activeEvents.end())
	{
		if ((*it)->finished()) {
			CUDAEvent* evt = *it;
			
			_eventPool.returnEvent(evt);
			finishedTasks.push_back(evt->getTask());
			
			it = _activeEvents.erase(it);
		} else {
			++it;
		}
	}
	
	return finishedTasks;
}

void CUDAComputePlace::preRunTask(Task *task)
{
	CUDADeviceData *taskData = (CUDADeviceData *) task->getDeviceData();
	
	task->setComputePlace(this);
	taskData->_stream = _streamPool.getStream();
}

void CUDAComputePlace::runTask(Task *task)
{
	CUDADeviceData *taskData = (CUDADeviceData *) task->getDeviceData();	
	
	nanos6_cuda_device_environment_t env;
	env.stream = taskData->_stream->getStream();
	
	task->body((void *) &env);
	
	CUDAEvent *event = _eventPool.getEvent();
	event->setTask(task);
	event->record();
	_activeEvents.push_back(event);
}

void CUDAComputePlace::postRunTask(Task *task)
{
	CUDADeviceData *taskData = (CUDADeviceData *) task->getDeviceData();	
	
	_streamPool.returnStream(taskData->_stream);
}

