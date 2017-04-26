#include "DeviceHierarchicalScheduler.hpp"

#include "../SchedulerGenerator.hpp"


DeviceHierarchicalScheduler::DeviceHierarchicalScheduler()
{
	_CPUScheduler = SchedulerGenerator::createDeviceScheduler();
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
