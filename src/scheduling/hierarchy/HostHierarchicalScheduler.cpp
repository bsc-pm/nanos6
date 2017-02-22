#include "HostHierarchicalScheduler.hpp"
#include "NUMAHierarchicalScheduler.hpp"

#include <cassert>


HostHierarchicalScheduler::HostHierarchicalScheduler()
{
	_NUMAScheduler = new NUMAHierarchicalScheduler();
}

HostHierarchicalScheduler::~HostHierarchicalScheduler()
{
	delete _NUMAScheduler;
}


SchedulerInterface *HostHierarchicalScheduler::getInstance()
{
	return _NUMAScheduler->getInstance();
}


ComputePlace * HostHierarchicalScheduler::addReadyTask(Task *task, ComputePlace *hardwarePlace, ReadyTaskHint hint)
{
	return _NUMAScheduler->addReadyTask(task, hardwarePlace, hint);
}


void HostHierarchicalScheduler::taskGetsUnblocked(Task *unblockedTask, ComputePlace *hardwarePlace)
{
	_NUMAScheduler->taskGetsUnblocked(unblockedTask, hardwarePlace);
}


Task *HostHierarchicalScheduler::getReadyTask(ComputePlace *hardwarePlace, Task *currentTask)
{
	return _NUMAScheduler->getReadyTask(hardwarePlace, currentTask);
}


ComputePlace *HostHierarchicalScheduler::getIdleComputePlace(bool force)
{
	return _NUMAScheduler->getIdleComputePlace(force);
}

