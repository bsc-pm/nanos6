/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include "HostUnsyncScheduler.hpp"
#include "scheduling/ready-queues/ReadyQueueDeque.hpp"
#include "scheduling/ready-queues/ReadyQueueMap.hpp"
#include "tasks/LoopGenerator.hpp"
#include "tasks/Task.hpp"
#include "tasks/Taskfor.hpp"

Task *HostUnsyncScheduler::getReadyTask(ComputePlace *computePlace)
{
	assert(computePlace != nullptr);
	Task *result = nullptr;
	Taskfor *groupTaskfor = nullptr;

	long cpuId = ((CPU *)computePlace)->getIndex();
	long groupId = (computePlace->getType() == nanos6_host_device) ? ((CPU *)computePlace)->getGroupId() : -1;
	long immediateSuccessorGroupId = groupId*2;

	// 1. Try to get work from the current group taskfor.
	if (groupId != -1) {
		if ((groupTaskfor = _groupSlots[groupId]) != nullptr) {

			groupTaskfor->notifyCollaboratorHasStarted();
			int myChunk = groupTaskfor->getNextChunk();
			if (myChunk <= 0) {
				_groupSlots[groupId] = nullptr;
				groupTaskfor->removedFromScheduler();
			}

			Taskfor *taskfor = computePlace->getPreallocatedTaskfor();
			// We are setting the chunk that the collaborator will execute in the preallocatedTaskfor.
			taskfor->setChunk(myChunk);
			return groupTaskfor;
		}
	}

	if (_enableImmediateSuccessor) {
		// 2. Try to get work from my immediateSuccessorTaskfors.
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

		// 3. Try to get work from my immediateSuccessorTasks.
		if (result == nullptr && _immediateSuccessorTasks[cpuId] != nullptr) {
			result = _immediateSuccessorTasks[cpuId];
			_immediateSuccessorTasks[cpuId] = nullptr;
		}
	}

	// 4. Check if there is work remaining in the ready queue.
	if (result == nullptr) {
		result = _readyTasks->getReadyTask(computePlace);
	}

	// 5. Try to get work from other immediateSuccessorTasks.
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

	// 6. Try to get work from other immediateSuccessorTasksfors.
	if (result == nullptr && _enableImmediateSuccessor) {
		for (size_t i = 0; i < _immediateSuccessorTaskfors.size(); i++) {
			if (_immediateSuccessorTaskfors[i] != nullptr) {
				result = _immediateSuccessorTaskfors[i];
				_immediateSuccessorTaskfors[i] = nullptr;
				break;
			}
		}
	}

	if (result == nullptr || !result->isTaskfor()) {
		assert(result == nullptr || result->isRunnable());
		return result;
	}

	assert(result->isTaskfor());
	assert(computePlace->getType() == nanos6_device_t::nanos6_host_device);

	Taskfor *taskfor = (Taskfor *) result;
	_groupSlots[groupId] = taskfor;
	taskfor->markAsScheduled();
	return getReadyTask(computePlace);
}


bool HostUnsyncScheduler::hasAvailableWork(ComputePlace *computePlace)
{
	if (computePlace != nullptr) {
		CPU *cpu = (CPU *) computePlace;
		long cpuId = cpu->getIndex();
		long groupId = (computePlace->getType() == nanos6_host_device) ? cpu->getGroupId() : -1;
		long immediateSuccessorGroupId = groupId * 2;

		// 1. Check if the CPU can participate in a taskfor
		if (groupId != -1 && _groupSlots[groupId] != nullptr) {
			return true;
		}

		if (_enableImmediateSuccessor) {
			// 2. Check for work in the CPU's immediateSuccessorTaskfors
			Task *currentIS1 = _immediateSuccessorTaskfors[immediateSuccessorGroupId];
			Task *currentIS2 = _immediateSuccessorTaskfors[immediateSuccessorGroupId + 1];
			if ((currentIS1 != nullptr) || (currentIS2 != nullptr)) {
				return true;
			}

			// 3. Check for work in the CPU's immediateSuccessorTasks
			if (_immediateSuccessorTasks[cpuId] != nullptr) {
				return true;
			}
		}
	}

	// 4. At this point we still don't find work, check the queue
	if (_readyTasks->getNumReadyTasks() != 0) {
		return true;
	}

	return false;
}
