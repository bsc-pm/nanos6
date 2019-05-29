/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018-2019 Barcelona Supercomputing Center (BSC)
*/

#include <random>

#include "ClusterRandomScheduler.hpp"
#include "scheduling/schedulers/HostHierarchicalScheduler.hpp"
#include "scheduling/SchedulerGenerator.hpp"
#include "system/RuntimeInfo.hpp"
#include "tasks/Task.hpp"

#include <ClusterManager.hpp>
#include <ExecutionWorkflow.hpp>

ClusterRandomScheduler::ClusterRandomScheduler()
{
	RuntimeInfo::addEntry("cluster-scheduler", "Cluster Scheduler", getName());
	_hostScheduler = new HostHierarchicalScheduler();
	_thisNode = ClusterManager::getCurrentClusterNode();
	_clusterSize = ClusterManager::clusterSize();
}

ClusterRandomScheduler::~ClusterRandomScheduler()
{
	delete _hostScheduler;
}

ComputePlace *ClusterRandomScheduler::addReadyTask(Task *task, ComputePlace *hardwarePlace, ReadyTaskHint hint, bool doGetIdle)
{
	if ((task->getParent() == nullptr) || (_clusterSize == 1) || task->isIf0() || task->isRemote() || task->getWorkflow() != nullptr) {
		return _hostScheduler->addReadyTask(task, hardwarePlace, hint, doGetIdle);
	}
	
	std::random_device rd;
	std::mt19937 eng(rd());
	std::uniform_int_distribution<> distr(0, _clusterSize - 1);
	
	ClusterNode *targetNode = ClusterManager::getClusterNode(distr(eng));
	assert(targetNode != nullptr);
	
	if (targetNode == _thisNode) {
		return _hostScheduler->addReadyTask(task, hardwarePlace, hint, doGetIdle);
	}
	
	ClusterMemoryNode *memoryNode = targetNode->getMemoryNode();
	assert(memoryNode != nullptr);
	
	ExecutionWorkflow::executeTask(task, targetNode, memoryNode);
	
	//! Offload task
	return nullptr;
}

Task *ClusterRandomScheduler::getReadyTask(ComputePlace *hardwarePlace, Task *currentTask, bool canMarkAsIdle, bool doWait)
{
	return _hostScheduler->getReadyTask(hardwarePlace, currentTask, canMarkAsIdle, doWait);
}

ComputePlace *ClusterRandomScheduler::getIdleComputePlace(bool force)
{
	return _hostScheduler->getIdleComputePlace(force);
}

void ClusterRandomScheduler::disableComputePlace(ComputePlace *hardwarePlace)
{
	_hostScheduler->disableComputePlace(hardwarePlace);
}

void ClusterRandomScheduler::enableComputePlace(ComputePlace *hardwarePlace)
{
	_hostScheduler->enableComputePlace(hardwarePlace);
}

bool ClusterRandomScheduler::requestPolling(ComputePlace *computePlace, polling_slot_t *pollingSlot)
{
	return _hostScheduler->requestPolling(computePlace, pollingSlot);
}

bool ClusterRandomScheduler::releasePolling(ComputePlace *computePlace, polling_slot_t *pollingSlot)
{
	return _hostScheduler->releasePolling(computePlace, pollingSlot);
}

std::string ClusterRandomScheduler::getName() const
{
	return "cluster-random";
}
