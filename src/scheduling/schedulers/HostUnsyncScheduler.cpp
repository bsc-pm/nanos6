/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2021 Barcelona Supercomputing Center (BSC)
*/

#include "HostUnsyncScheduler.hpp"
#include "scheduling/ready-queues/DeadlineQueue.hpp"
#include "scheduling/ready-queues/ReadyQueueDeque.hpp"
#include "scheduling/ready-queues/ReadyQueueMap.hpp"
#include "tasks/LoopGenerator.hpp"
#include "tasks/Task.hpp"
#include "tasks/Taskfor.hpp"

Task *HostUnsyncScheduler::getReadyTask(ComputePlace *computePlace, bool &hasIncompatibleWork)
{
	assert(computePlace != nullptr);
	assert(_deadlineTasks != nullptr);

	Task *result = nullptr;
	Taskfor *groupTaskfor = nullptr;

	long cpuId = computePlace->getIndex();
	long groupId = ((CPU *)computePlace)->getGroupId();

	hasIncompatibleWork = false;

	// 1. Try to get a task with a satisfied deadline
	result = _deadlineTasks->getReadyTask(computePlace);
	if (result != nullptr) {
		return result;
	}

	// 2. Try to get work from the current group taskfor
	if (groupId != -1) {
		if ((groupTaskfor = _groupSlots[groupId]) != nullptr) {

			groupTaskfor->notifyCollaboratorHasStarted();
			bool remove = false;
			int myChunk = groupTaskfor->getNextChunk(cpuId, &remove);
			if (remove) {
				_groupSlots[groupId] = nullptr;
				groupTaskfor->removedFromScheduler();
			}

			// We are setting the chunk that the collaborator will execute in the preallocatedTaskfor
			Taskfor *taskfor = computePlace->getPreallocatedTaskfor();
			taskfor->setChunk(myChunk);
			return groupTaskfor;
		}
	}

	// 3. Check if there is work remaining in the ready queue
	if (result == nullptr) {
		result = regularGetReadyTask(computePlace);
	}

	if (result == nullptr) {
		// 4. If there is a hidden Taskfor in any of the slots not accessible
		// to this computePlace, alert about it through the bool
		for (int i = 0; i < (int) _groupSlots.size(); ++i) {
			if (i != groupId && _groupSlots[i] != nullptr) {
				hasIncompatibleWork = true;
			}
		}
		return result;
	}

	if (!result->isTaskforSource()) {
		return result;
	}

	assert(result->isTaskfor());
	assert(computePlace->getType() == nanos6_host_device);

	Taskfor *taskfor = (Taskfor *) result;
	_groupSlots[groupId] = taskfor;
	taskfor->markAsScheduled();
	return getReadyTask(computePlace, hasIncompatibleWork);
}
