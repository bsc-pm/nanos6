#include "HostHierarchicalScheduler.hpp"

#include "../SchedulerGenerator.hpp"


HostHierarchicalScheduler::HostHierarchicalScheduler()
{
	_NUMAScheduler = SchedulerGenerator::createNUMAScheduler();
}

HostHierarchicalScheduler::~HostHierarchicalScheduler()
{
	delete _NUMAScheduler;
}


ComputePlace * HostHierarchicalScheduler::addReadyTask(Task *task, ComputePlace *hardwarePlace, ReadyTaskHint hint, bool doGetIdle)
{
	return _NUMAScheduler->addReadyTask(task, hardwarePlace, hint, doGetIdle);
}


void HostHierarchicalScheduler::taskGetsUnblocked(Task *unblockedTask, ComputePlace *hardwarePlace)
{
	_NUMAScheduler->taskGetsUnblocked(unblockedTask, hardwarePlace);
}


Task *HostHierarchicalScheduler::getReadyTask(ComputePlace *hardwarePlace, Task *currentTask, bool canMarkAsIdle)
{
	return _NUMAScheduler->getReadyTask(hardwarePlace, currentTask, canMarkAsIdle);
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
