/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include <vector>

#include "ClusterLocalityScheduler.hpp"
#include "memory/directory/Directory.hpp"
#include "system/RuntimeInfo.hpp"
#include "tasks/Task.hpp"

#include <ClusterManager.hpp>
#include <DataAccessRegistrationImplementation.hpp>
#include <ExecutionWorkflow.hpp>
#include <VirtualMemoryManagement.hpp>

void ClusterLocalityScheduler::addReadyTask(Task *task, ComputePlace *computePlace,
		ReadyTaskHint hint)
{
	//! We do not offload spawned functions, if0 tasks, remote task
	//! and tasks that already have an ExecutionWorkflow created for
	//! them
	if ((task->isSpawned() || task->isIf0() || task->isRemote() ||
		task->getWorkflow() != nullptr)) {
		SchedulerInterface::addReadyTask(task, computePlace, hint);
		return;
	}
	
	std::vector<size_t> bytes(_clusterSize, 0);
	bool canBeOffloaded = true;
	DataAccessRegistration::processAllDataAccesses(task,
		[&](DataAccessRegion region, __attribute__((unused))DataAccessType type,
			__attribute__((unused))bool isWeak, MemoryPlace const *location) -> bool {
			if (location == nullptr) {
				assert(isWeak);
				location = Directory::getDirectoryMemoryPlace();
			}
			
			if (!VirtualMemoryManagement::isClusterMemory(region)) {
				canBeOffloaded = false;
				return false;
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
					
					DataAccessRegion subregion =
						region.intersect(entry->getAccessRegion());
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
	
	if (!canBeOffloaded) {
		SchedulerInterface::addReadyTask(task, computePlace, hint);
		return;
	}
	
	assert(!bytes.empty());
	std::vector<size_t>::iterator it = bytes.begin();
	size_t nodeId = std::distance(it, std::max_element(it, it + _clusterSize));
	
	ClusterNode *targetNode = ClusterManager::getClusterNode(nodeId);
	assert(targetNode != nullptr);
	if (targetNode == _thisNode) {
		SchedulerInterface::addReadyTask(task, computePlace, hint);
		return;
	}
	
	ClusterMemoryNode *memoryNode = targetNode->getMemoryNode();
	assert(memoryNode != nullptr);
	ExecutionWorkflow::executeTask(task, targetNode, memoryNode);
}
