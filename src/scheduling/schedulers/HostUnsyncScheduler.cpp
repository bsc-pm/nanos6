/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "HostUnsyncScheduler.hpp"
#include "scheduling/ready-queues/ReadyQueueDeque.hpp"
#include "scheduling/ready-queues/ReadyQueueMap.hpp"
#include "tasks/Task.hpp"
#include "tasks/Taskfor.hpp"
#include "tasks/TaskforGenerator.hpp"

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
			assert(groupTaskfor->hasPendingIterations());
			
			groupTaskfor->notifyCollaboratorHasStarted();
			TaskforInfo::bounds_t bounds;
			bool clearSlot = groupTaskfor->getChunks(bounds);
			if (clearSlot) {
				_groupSlots[groupId] = nullptr;
			}
			
			Taskfor *collaborator = TaskforGenerator::createCollaborator(groupTaskfor, bounds, computePlace);
			
			assert(collaborator->isRunnable());
			return collaborator;
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
	
	_groupSlots[groupId] = (Taskfor *) result;
	return getReadyTask(computePlace);
}
