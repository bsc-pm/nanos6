/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include "HostUnsyncScheduler.hpp"
#include "scheduling/ready-queues/DeadlineQueue.hpp"
#include "scheduling/ready-queues/ReadyQueueDeque.hpp"
#include "scheduling/ready-queues/ReadyQueueMap.hpp"
#include "tasks/LoopGenerator.hpp"
#include "tasks/Task.hpp"
#include "tasks/Taskfor.hpp"

Task *HostUnsyncScheduler::getReadyTask(ComputePlace *computePlace)
{
	assert(computePlace != nullptr);
	assert(_deadlineTasks != nullptr);

	Task *result = nullptr;
	Taskfor *groupTaskfor = nullptr;

	long cpuId = computePlace->getIndex();
	long groupId = ((CPU *)computePlace)->getGroupId();
	long immediateSuccessorGroupId = groupId*2;

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

			Taskfor *taskfor = computePlace->getPreallocatedTaskfor();
			// We are setting the chunk that the collaborator will execute in the preallocatedTaskfor
			taskfor->setChunk(myChunk);
			return groupTaskfor;
		}
	}

	if (_enableImmediateSuccessor) {
		// 3. Try to get work from my immediateSuccessorTaskfors
		Task *currentImmediateSuccessor1 = _immediateSuccessorTaskfors[immediateSuccessorGroupId];
		Task *currentImmediateSuccessor2 = _immediateSuccessorTaskfors[immediateSuccessorGroupId+1];
		if (currentImmediateSuccessor1 != nullptr) {
			assert(currentImmediateSuccessor1->isTaskfor());
			result = currentImmediateSuccessor1;
			_immediateSuccessorTaskfors[immediateSuccessorGroupId] = nullptr;
		}
		else if (currentImmediateSuccessor2 != nullptr) {
			assert(currentImmediateSuccessor2->isTaskfor());
			result = currentImmediateSuccessor2;
			_immediateSuccessorTaskfors[immediateSuccessorGroupId+1] = nullptr;
		}

		// 4. Try to get work from my immediateSuccessorTasks
		if (result == nullptr && _immediateSuccessorTasks[cpuId] != nullptr) {
			result = _immediateSuccessorTasks[cpuId];
			_immediateSuccessorTasks[cpuId] = nullptr;
		}
	}

	// 5. Check if there is work remaining in the ready queue
	if (result == nullptr) {
		result = regularGetReadyTask(computePlace);
	}

	// 6. Try to get work from other immediateSuccessorTasks
	if (result == nullptr && _enableImmediateSuccessor) {
		for (size_t i = 0; i < _immediateSuccessorTasks.size(); i++) {
			if (_immediateSuccessorTasks[i] != nullptr) {
				result = _immediateSuccessorTasks[i];
				assert(!result->isTaskfor());
				_immediateSuccessorTasks[i] = nullptr;
				break;
			}
		}
	}

	// 7. Try to get work from other immediateSuccessorTasksfors
	if (result == nullptr && _enableImmediateSuccessor) {
		for (size_t i = 0; i < _immediateSuccessorTaskfors.size(); i++) {
			if (_immediateSuccessorTaskfors[i] != nullptr) {
				result = _immediateSuccessorTaskfors[i];
				_immediateSuccessorTaskfors[i] = nullptr;
				break;
			}
		}
	}

	if (result == nullptr || !result->isTaskforSource()) {
		return result;
	}

	assert(result->isTaskfor());
	assert(computePlace->getType() == nanos6_host_device);

	Taskfor *taskfor = (Taskfor *) result;
	_groupSlots[groupId] = taskfor;
	taskfor->markAsScheduled();
	return getReadyTask(computePlace);
}
