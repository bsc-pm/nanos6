/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "system/RuntimeInfo.hpp"

#include "HostHierarchicalScheduler.hpp"

#include "../SchedulerGenerator.hpp"

#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"

#include <config.h>

HostHierarchicalScheduler::HostHierarchicalScheduler()
{
	_NUMAScheduler = SchedulerGenerator::createNUMAScheduler();
	RuntimeInfo::addEntry("numa-scheduler", "NUMA Scheduler", _NUMAScheduler->getName());

#ifdef USE_CUDA	
	_CUDAScheduler = SchedulerGenerator::createDeviceScheduler(0, nanos6_device_t::nanos6_cuda_device);
	RuntimeInfo::addEntry("cuda-scheduler", "CUDA Scheduler", _CUDAScheduler->getName());
#endif //USE_CUDA
}

HostHierarchicalScheduler::~HostHierarchicalScheduler()
{
	delete _NUMAScheduler;

#ifdef USE_CUDA
	delete _CUDAScheduler;
#endif //USE_CUDA
}


ComputePlace * HostHierarchicalScheduler::addReadyTask(Task *task, ComputePlace *hardwarePlace, ReadyTaskHint hint, bool doGetIdle)
{
	switch (task->getDeviceType()) {	
		case nanos6_device_t::nanos6_host_device:
			return _NUMAScheduler->addReadyTask(task, hardwarePlace, hint, doGetIdle);
			break;
#ifdef USE_CUDA
		case nanos6_device_t::nanos6_cuda_device:
			return _CUDAScheduler->addReadyTask(task, hardwarePlace, hint, doGetIdle);
			break;
#endif //USE_CUDA
		default:
			std::cerr << "Task type " << task->getDeviceType() << "is not supported, defaulting to Host task" << std::endl;
			return _NUMAScheduler->addReadyTask(task, hardwarePlace, hint, doGetIdle);
			break;
	}
}


void HostHierarchicalScheduler::taskGetsUnblocked(Task *unblockedTask, ComputePlace *hardwarePlace)
{
	_NUMAScheduler->taskGetsUnblocked(unblockedTask, hardwarePlace);
}


Task *HostHierarchicalScheduler::getReadyTask(ComputePlace *hardwarePlace, Task *currentTask, bool canMarkAsIdle)
{
	switch (hardwarePlace->getType()) {
		case nanos6_device_t::nanos6_host_device:
			return _NUMAScheduler->getReadyTask(hardwarePlace, currentTask, canMarkAsIdle);
			break;
#ifdef USE_CUDA
		case nanos6_device_t::nanos6_cuda_device:
			return _CUDAScheduler->getReadyTask(hardwarePlace, currentTask, canMarkAsIdle);
			break;
#endif //USE_CUDA
		default:
			std::cerr << "Device type " << hardwarePlace->getType() << "is not supported, defaulting to CPU" << std::endl;
			return _NUMAScheduler->getReadyTask(hardwarePlace, currentTask, canMarkAsIdle);
			break;
	}
}


ComputePlace *HostHierarchicalScheduler::getIdleComputePlace(bool force)
{
	return _NUMAScheduler->getIdleComputePlace(force);
}

void HostHierarchicalScheduler::disableComputePlace(ComputePlace *hardwarePlace)
{
	_NUMAScheduler->disableComputePlace(hardwarePlace);
}

void HostHierarchicalScheduler::enableComputePlace(ComputePlace *hardwarePlace)
{
	_NUMAScheduler->enableComputePlace(hardwarePlace);
}

bool HostHierarchicalScheduler::requestPolling(ComputePlace *computePlace, polling_slot_t *pollingSlot)
{
	return _NUMAScheduler->requestPolling(computePlace, pollingSlot);
}

bool HostHierarchicalScheduler::releasePolling(ComputePlace *computePlace, polling_slot_t *pollingSlot)
{
	return _NUMAScheduler->releasePolling(computePlace, pollingSlot);
}


std::string HostHierarchicalScheduler::getName() const
{
	return "host-hierarchical";
}
