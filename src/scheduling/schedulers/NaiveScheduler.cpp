/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "NaiveScheduler.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "executors/threads/TaskFinalization.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/CPUManager.hpp"
#include "hardware/places/CPUPlace.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"

#include <DataAccessRegistration.hpp>

#include <algorithm>
#include <cassert>
#include <mutex>

NaiveScheduler::NaiveScheduler(__attribute__((unused)) int numaNodeIndex)
{
}

NaiveScheduler::~NaiveScheduler()
{
}


Task *NaiveScheduler::getReplacementTask(__attribute__((unused)) CPU *computePlace)
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


ComputePlace * NaiveScheduler::addReadyTask(Task *task, __attribute__((unused)) ComputePlace *computePlace, __attribute__((unused)) ReadyTaskHint hint, bool doGetIdle)
{
	std::lock_guard<SpinLock> guard(_globalLock);
	_readyTasks.push_front(task);
	
	if (doGetIdle) {
		return CPUManager::getIdleCPU();
	} else {
		return nullptr;
	}
}


void NaiveScheduler::taskGetsUnblocked(Task *unblockedTask, __attribute__((unused)) ComputePlace *computePlace)
{
	std::lock_guard<SpinLock> guard(_globalLock);
	_unblockedTasks.push_front(unblockedTask);
}


Task *NaiveScheduler::getReadyTask(ComputePlace *computePlace, __attribute__((unused)) Task *currentTask, bool canMarkAsIdle)
{
	Task *task = nullptr;
	
	std::lock_guard<SpinLock> guard(_globalLock);
	
	// Try to get an unblocked task
	task = getReplacementTask((CPU *) computePlace);
	if (task != nullptr) {
		return task;
	}
	
	if (!_readyTasks.empty() ) {
		// Get the first ready task
		task = _readyTasks.front();
		_readyTasks.pop_front();
	
		assert(task != nullptr);
		
		return task;
	
	}
	
	if (canMarkAsIdle) {
		CPUManager::cpuBecomesIdle((CPU *)computePlace);
	}
	
	return nullptr;
}


ComputePlace *NaiveScheduler::getIdleComputePlace(bool force)
{
	std::lock_guard<SpinLock> guard(_globalLock);
	if (force || !_readyTasks.empty() || !_unblockedTasks.empty()) {
		return CPUManager::getIdleCPU();
	} else {
		return nullptr;
	}
}


std::string NaiveScheduler::getName() const
{
	return "naive";
}
