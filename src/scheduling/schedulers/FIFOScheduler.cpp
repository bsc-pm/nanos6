/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "FIFOScheduler.hpp"
#include "executors/threads/CPUManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "executors/threads/TaskFinalization.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "hardware/places/CPUPlace.hpp"
#include "scheduling/TaskloopSchedulingPolicy.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"
#include "tasks/Taskloop.hpp"
#include "tasks/TaskloopGenerator.hpp"

#include <DataAccessRegistration.hpp>

#include <algorithm>
#include <cassert>
#include <mutex>

FIFOScheduler::FIFOScheduler(__attribute__((unused)) int numaNodeIndex)
{
}

FIFOScheduler::~FIFOScheduler()
{
}


Task *FIFOScheduler::getReplacementTask(__attribute__((unused)) CPU *computePlace)
{
	if (!_unblockedTasks.empty()) {
		Task *replacementTask = _unblockedTasks.front();
		_unblockedTasks.pop_front();
		
		assert(replacementTask != nullptr);
		
		return replacementTask;
	} else {
		return nullptr;
	}
}


ComputePlace * FIFOScheduler::addReadyTask(Task *task, __attribute__((unused)) ComputePlace *computePlace, __attribute__((unused)) ReadyTaskHint hint, bool doGetIdle)
{
	FatalErrorHandler::failIf(task->getDeviceType() != nanos6_device_t::nanos6_host_device, "Device tasks not supported by this scheduler");
	
	std::lock_guard<SpinLock> guard(_globalLock);
	
	if (hint == UNBLOCKED_TASK_HINT) {
		_unblockedTasks.push_back(task);
	} else {
		_readyTasks.push_back(task);
	}
	
	if (doGetIdle) {
		return CPUManager::getIdleCPU();
	} else {
		return nullptr;
	}
}


Task *FIFOScheduler::getReadyTask(ComputePlace *computePlace, __attribute__((unused)) Task *currentTask, bool canMarkAsIdle, __attribute__((unused)) bool doWait)
{
	if (computePlace->getType() != nanos6_device_t::nanos6_host_device) { 
		return nullptr;
	}
	
	Task *task = nullptr;
	bool workAssigned = false;
	std::vector<Taskloop *> completeTaskloops;
	
	{
		std::lock_guard<SpinLock> guard(_globalLock);
		
		// Try to get an unblocked task
		task = getReplacementTask((CPU *) computePlace);
		if (task != nullptr) {
			return task;
		}
		
		while (!_readyTasks.empty() && !workAssigned) {
			// Get the first ready task
			task = _readyTasks.front();
			assert(task != nullptr);
			
			if (!task->isTaskloop()) {
				_readyTasks.pop_front();
				workAssigned = true;
				break;
			}
			
			Taskloop *taskloop = (Taskloop *)task;
			workAssigned = taskloop->hasPendingIterations();
			
			if (workAssigned) {
				if (TaskloopSchedulingPolicy::isRequeueEnabled()) {
					_readyTasks.pop_front();
					_readyTasks.push_back(taskloop);
				}
				taskloop->notifyCollaboratorHasStarted();
				break;
			}
			
			_readyTasks.pop_front();
			completeTaskloops.push_back(taskloop);
		}
	}
	
	bool shouldRecheckUnblockedTasks = false;
	for (Taskloop *taskloop : completeTaskloops) {
		if (taskloop->markAsFinished(computePlace)) {
			DataAccessRegistration::unregisterTaskDataAccesses(
				taskloop,
				computePlace,
				computePlace->getDependencyData()
			);
			if (taskloop->markAsReleased()) {
				TaskFinalization::disposeOrUnblockTask(taskloop, computePlace);
			}
			shouldRecheckUnblockedTasks = true;
		}
	}
	
	if (workAssigned) {
		assert(task != nullptr);
		
		if (task->isTaskloop()) {
			return TaskloopGenerator::createCollaborator((Taskloop *)task);
		}
		
		return task;
	}
	
	if (shouldRecheckUnblockedTasks) {
		return Scheduler::getReadyTask(computePlace, currentTask);
	}
	
	if (canMarkAsIdle) {
		CPUManager::cpuBecomesIdle((CPU *)computePlace);
	}
	
	return nullptr;
}


ComputePlace *FIFOScheduler::getIdleComputePlace(bool force)
{
	std::lock_guard<SpinLock> guard(_globalLock);
	if (force || !_readyTasks.empty() || !_unblockedTasks.empty()) {
		return CPUManager::getIdleCPU();
	} else {
		return nullptr;
	}
}


std::string FIFOScheduler::getName() const
{
	return "fifo";
}
