/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#include "HostUnsyncScheduler.hpp"

#include "scheduling/ready-queues/ReadyQueueDeque.hpp"
#include "scheduling/ready-queues/ReadyQueueMap.hpp"
#include "tasks/Task.hpp"
#include "tasks/Taskloop.hpp"
#include "tasks/TaskloopGenerator.hpp"

Task *HostUnsyncScheduler::getReadyTask(ComputePlace *computePlace)
{
	Task *task = nullptr;
	
	// 1. Check if there is an active taskloop.
	if (_currentTaskloop != nullptr) {
		Taskloop *taskloop = _currentTaskloop;
		bool pendingWork = _currentTaskloop->hasPendingIterations();
		taskloop->notifyCollaboratorHasStarted();
		if (!pendingWork) {
			_currentTaskloop = nullptr;
			__attribute__((unused)) bool finished = taskloop->markAsFinished(computePlace);
			assert(!finished);
		}
		return TaskloopGenerator::createCollaborator(taskloop);
	}
	
	// 2. Check if there is an immediate successor.
	if (_enableImmediateSuccessor && computePlace != nullptr) {
		size_t immediateSuccessorId = computePlace->getIndex();
		if (_immediateSuccessorTasks[immediateSuccessorId] != nullptr) {
			task = _immediateSuccessorTasks[immediateSuccessorId];
			_immediateSuccessorTasks[immediateSuccessorId] = nullptr;
			assert(!task->isTaskloop());
			return task;
		}
	}
	
	// 3. Check if there is work remaining in the ready queue.
	task = _readyTasks->getReadyTask(computePlace);
	
	 //4. Try to get work from other immediateSuccessorTasks.
	if (task == nullptr && _enableImmediateSuccessor) {
		for (size_t i = 0; i < _immediateSuccessorTasks.size(); i++) {
			if (_immediateSuccessorTasks[i] != nullptr) {
				task = _immediateSuccessorTasks[i];
				assert(!task->isTaskloop());
				_immediateSuccessorTasks[i] = nullptr;
				break;
			}
		}
	}
	
	if (task != nullptr && task->isTaskloop() && !task->isRunnable()) {
		// If a taskloop is non-runnable, it means that it is a "source" taskloop that must be executed by collaborators.
		// Otherwise, it is already a collaborator that has been blocked for some reason, and now it must be executed as a normal task.
		Taskloop *taskloop = (Taskloop *) task;
		bool pendingWork = taskloop->hasPendingIterations();
		taskloop->notifyCollaboratorHasStarted();
		if (pendingWork) {
			assert(_currentTaskloop == nullptr);
			_currentTaskloop = taskloop;
		}
		else {
			__attribute__((unused)) bool finished = taskloop->markAsFinished(computePlace);
			assert(!finished);
		}
		return TaskloopGenerator::createCollaborator(taskloop);
	}
	
	return task;
}
