#include "system/RuntimeInfo.hpp"

#include "DeviceHierarchicalScheduler.hpp"

#include "../SchedulerGenerator.hpp"

#include <sstream>


DeviceHierarchicalScheduler::DeviceHierarchicalScheduler(int nodeIndex)
{
	_CPUScheduler = SchedulerGenerator::createDeviceScheduler(nodeIndex);
	
	std::ostringstream oss, oss2;
	if (nodeIndex != -1) {
		oss << "numa-node-" << nodeIndex + 1 << "-";
		oss2 << "NUMA Node " << nodeIndex + 1 << " ";
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


void DeviceHierarchicalScheduler::taskGetsUnblocked(Task *unblockedTask, ComputePlace *hardwarePlace)
{
	_CPUScheduler->taskGetsUnblocked(unblockedTask, hardwarePlace);
}


Task *DeviceHierarchicalScheduler::getReadyTask(ComputePlace *hardwarePlace, Task *currentTask, bool canMarkAsIdle)
{
	return _CPUScheduler->getReadyTask(hardwarePlace, currentTask, canMarkAsIdle);
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

bool DeviceHierarchicalScheduler::requestPolling(ComputePlace *computePlace, polling_slot_t *pollingSlot)
{
	return _CPUScheduler->requestPolling(computePlace, pollingSlot);
}

bool DeviceHierarchicalScheduler::releasePolling(ComputePlace *computePlace, polling_slot_t *pollingSlot)
{
	return _CPUScheduler->releasePolling(computePlace, pollingSlot);
}


std::string DeviceHierarchicalScheduler::getName() const
{
	return "device-hierachical";
}
