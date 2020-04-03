/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include <random>

#include "ClusterRandomScheduler.hpp"

#include <ClusterManager.hpp>
#include <DataAccessRegistrationImplementation.hpp>
#include <ExecutionWorkflow.hpp>
#include <VirtualMemoryManagement.hpp>

void ClusterRandomScheduler::addReadyTask(Task *task, ComputePlace *computePlace,
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

	bool canBeOffloaded = true;
	DataAccessRegistration::processAllDataAccesses(task,
		[&](DataAccessRegion region, DataAccessType, bool,
						MemoryPlace const *) -> bool {
			if (!VirtualMemoryManagement::isClusterMemory(region)) {
				canBeOffloaded = false;
				return false;
			}

			return true;
		}
	);

	if (!canBeOffloaded) {
		SchedulerInterface::addReadyTask(task, computePlace, hint);
		return;
	}

	std::random_device rd;
	std::mt19937 eng(rd());
	std::uniform_int_distribution<> distr(0, _clusterSize - 1);

	ClusterNode *targetNode = ClusterManager::getClusterNode(distr(eng));
	assert(targetNode != nullptr);

	if (targetNode == _thisNode) {
		//! Execute task locally
		SchedulerInterface::addReadyTask(task, computePlace, hint);
		return;
	}

	ClusterMemoryNode *memoryNode = targetNode->getMemoryNode();
	assert(memoryNode != nullptr);

	//! Offload task
	ExecutionWorkflow::executeTask(task, targetNode, memoryNode);
}
