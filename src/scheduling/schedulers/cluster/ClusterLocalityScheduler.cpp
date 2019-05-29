/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018-2019 Barcelona Supercomputing Center (BSC)
*/

#include "ClusterLocalityScheduler.hpp"
#include "memory/directory/Directory.hpp"
#include "scheduling/schedulers/HostHierarchicalScheduler.hpp"
#include "system/RuntimeInfo.hpp"
#include "tasks/Task.hpp"

#include <ClusterManager.hpp>
#include <DataAccessRegistrationImplementation.hpp>
#include <ExecutionWorkflow.hpp>

ClusterLocalityScheduler::ClusterLocalityScheduler()
{
	RuntimeInfo::addEntry("cluster-scheduler", "Cluster Scheduler", getName());
	_hostScheduler = new HostHierarchicalScheduler();
	_thisNode = ClusterManager::getCurrentClusterNode();
	_clusterSize = ClusterManager::clusterSize();
}

ClusterLocalityScheduler::~ClusterLocalityScheduler()
{
	delete _hostScheduler;
}

ComputePlace *ClusterLocalityScheduler::addReadyTask(Task *task, ComputePlace *hardwarePlace, ReadyTaskHint hint, bool doGetIdle)
{
	if (task->isSpawned() || (_clusterSize == 1) || task->isIf0() || task->isRemote() || task->getWorkflow() != nullptr) {
		return _hostScheduler->addReadyTask(task, hardwarePlace, hint, doGetIdle);
	}
	
	std::vector<size_t> bytes(_clusterSize, 0);
	
	DataAccessRegistration::processAllDataAccesses(task,
		[&](DataAccessRegion region, __attribute__((unused))DataAccessType type,
			__attribute__((unused))bool isWeak, MemoryPlace const *location) -> bool {
			if (location == nullptr) {
				assert(isWeak);
				
				location = Directory::getDirectoryMemoryPlace();
			}
			
			if (Directory::isDirectoryMemoryPlace(location)) {
				Directory::HomeNodesArray *homeNodes =
					Directory::find(region);
				
				for (const auto &entry : *homeNodes) {
					location = entry->getHomeNode();
					
					size_t nodeId;
					if (location->getType() == nanos6_host_device) {
						nodeId = _thisNode->getIndex();
					} else {
						nodeId = location->getIndex();
					}
					
					DataAccessRegion subregion = region.intersect(entry->getAccessRegion());
					bytes[nodeId] += subregion.getSize();
				}
				
				delete homeNodes;
				
			} else {
				size_t nodeId;
				if (location->getType() == nanos6_host_device) {
					nodeId = _thisNode->getIndex();
				} else {
					nodeId = location->getIndex();
				}
				
				bytes[nodeId] += region.getSize();
			}
			
			return true;
		}
	);
	
	assert(!bytes.empty());
	std::vector<size_t>::iterator it = bytes.begin();
	size_t nodeId = std::distance(it, std::max_element(it, it + _clusterSize));
	
	ClusterNode *targetNode = ClusterManager::getClusterNode(nodeId);
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

Task *ClusterLocalityScheduler::getReadyTask(ComputePlace *hardwarePlace, Task *currentTask, bool canMarkAsIdle, bool doWait)
{
	return _hostScheduler->getReadyTask(hardwarePlace, currentTask, canMarkAsIdle, doWait);
}

ComputePlace *ClusterLocalityScheduler::getIdleComputePlace(bool force)
{
	return _hostScheduler->getIdleComputePlace(force);
}

void ClusterLocalityScheduler::disableComputePlace(ComputePlace *hardwarePlace)
{
	_hostScheduler->disableComputePlace(hardwarePlace);
}

void ClusterLocalityScheduler::enableComputePlace(ComputePlace *hardwarePlace)
{
	_hostScheduler->enableComputePlace(hardwarePlace);
}

bool ClusterLocalityScheduler::requestPolling(ComputePlace *computePlace, polling_slot_t *pollingSlot)
{
	return _hostScheduler->requestPolling(computePlace, pollingSlot);
}

bool ClusterLocalityScheduler::releasePolling(ComputePlace *computePlace, polling_slot_t *pollingSlot)
{
	return _hostScheduler->releasePolling(computePlace, pollingSlot);
}

std::string ClusterLocalityScheduler::getName() const
{
	return "cluster-locality";
}
