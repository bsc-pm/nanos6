/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "system/RuntimeInfo.hpp"

#include "DeviceHierarchicalScheduler.hpp"

#include "../SchedulerGenerator.hpp"

#include <sstream>


DeviceHierarchicalScheduler::DeviceHierarchicalScheduler(int numaNodeIndex)
{
	_CPUScheduler = SchedulerGenerator::createDeviceScheduler(numaNodeIndex, nanos6_device_t::nanos6_host_device);
	
	std::ostringstream oss, oss2;
	if (numaNodeIndex != -1) {
		oss << "numa-node-" << numaNodeIndex + 1 << "-";
		oss2 << "NUMA Node " << numaNodeIndex + 1 << " ";
	}
	oss << "cpu-scheduler";
	oss2 << "CPU Scheduler";
	RuntimeInfo::addEntry(oss.str(), oss2.str(), _CPUScheduler->getName());
}

DeviceHierarchicalScheduler::~DeviceHierarchicalScheduler()
{
	delete _CPUScheduler;
}


ComputePlace * DeviceHierarchicalScheduler::addReadyTask(Task *task, ComputePlace *hardwarePlace, ReadyTaskHint hint, bool doGetIdle)
{
	return _CPUScheduler->addReadyTask(task, hardwarePlace, hint, doGetIdle);
}


Task *DeviceHierarchicalScheduler::getReadyTask(ComputePlace *hardwarePlace, Task *currentTask, bool canMarkAsIdle, bool doWait)
{
	return _CPUScheduler->getReadyTask(hardwarePlace, currentTask, canMarkAsIdle, doWait);
}


ComputePlace *DeviceHierarchicalScheduler::getIdleComputePlace(bool force)
{
	return _CPUScheduler->getIdleComputePlace(force);
}

void DeviceHierarchicalScheduler::disableComputePlace(ComputePlace *hardwarePlace)
{
	_CPUScheduler->disableComputePlace(hardwarePlace);
}

void DeviceHierarchicalScheduler::enableComputePlace(ComputePlace *hardwarePlace)
{
	_CPUScheduler->enableComputePlace(hardwarePlace);
}

bool DeviceHierarchicalScheduler::requestPolling(ComputePlace *computePlace, polling_slot_t *pollingSlot, bool canMarkAsIdle)
{
	return _CPUScheduler->requestPolling(computePlace, pollingSlot, canMarkAsIdle);
}

bool DeviceHierarchicalScheduler::releasePolling(ComputePlace *computePlace, polling_slot_t *pollingSlot, bool canMarkAsIdle)
{
	return _CPUScheduler->releasePolling(computePlace, pollingSlot, canMarkAsIdle);
}


std::string DeviceHierarchicalScheduler::getName() const
{
	return "device-hierachical";
}
