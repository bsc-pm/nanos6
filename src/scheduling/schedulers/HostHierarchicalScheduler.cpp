#include "HostHierarchicalScheduler.hpp"

#include "../SchedulerGenerator.hpp"

#include <cassert>


HostHierarchicalScheduler::HostHierarchicalScheduler()
{
	_NUMAScheduler = SchedulerGenerator::createNUMAScheduler();
}

HostHierarchicalScheduler::~HostHierarchicalScheduler()
{
	delete _NUMAScheduler;
}


ComputePlace * HostHierarchicalScheduler::addReadyTask(Task *task, ComputePlace *hardwarePlace, ReadyTaskHint hint)
{
	return _NUMAScheduler->addReadyTask(task, hardwarePlace, hint);
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
