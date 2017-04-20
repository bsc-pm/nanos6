#include "ImmediateSuccessorScheduler.hpp"
#include "executors/threads/CPUManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "hardware/places/CPUPlace.hpp"
#include "tasks/Task.hpp"

#include <algorithm>
#include <cassert>
#include <mutex>

ImmediateSuccessorScheduler::ImmediateSuccessorScheduler()
{
}

ImmediateSuccessorScheduler::~ImmediateSuccessorScheduler()
{
}


Task *ImmediateSuccessorScheduler::getReplacementTask(__attribute__((unused)) CPU *computePlace)
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


ComputePlace * ImmediateSuccessorScheduler::addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint)
{
	// The following condition is only needed for the "main" task, that is added by something that is not a hardware place and thus should end up in a queue
	if (computePlace != nullptr) {
		if ((hint != CHILD_TASK_HINT) && (computePlace->_schedulerData == nullptr)) {
			computePlace->_schedulerData = task;
			return nullptr;
		}
	}
	
	std::lock_guard<SpinLock> guard(_globalLock);
	_readyTasks.push_front(task);
	
	return CPUManager::getIdleCPU();
}


void ImmediateSuccessorScheduler::taskGetsUnblocked(Task *unblockedTask, __attribute__((unused)) ComputePlace *computePlace)
{
	std::lock_guard<SpinLock> guard(_globalLock);
	_unblockedTasks.push_front(unblockedTask);
}


Task *ImmediateSuccessorScheduler::getReadyTask(ComputePlace *computePlace, __attribute__((unused)) Task *currentTask, bool canMarkAsIdle)
{
	Task *task = nullptr;
	
	// 1. Get the immediate successor
	if (computePlace->_schedulerData != nullptr) {
		task = (Task *) computePlace->_schedulerData;
		computePlace->_schedulerData = nullptr;
		return task;
	}
	
	std::lock_guard<SpinLock> guard(_globalLock);
	
	// 2. Get an unblocked task
	task = getReplacementTask((CPU *) computePlace);
	if (task != nullptr) {
		return task;
	}
	
	// 3. Or get a ready task
	if (!_readyTasks.empty()) {
		task = _readyTasks.front();
		_readyTasks.pop_front();
		
		assert(task != nullptr);
		
		return task;
	}
	
	// 4. Or mark the CPU as idle
	if (canMarkAsIdle) {
		CPUManager::cpuBecomesIdle((CPU *) computePlace);
	}
	
	return nullptr;
}


ComputePlace *ImmediateSuccessorScheduler::getIdleComputePlace(bool force)
{
	std::lock_guard<SpinLock> guard(_globalLock);
	if (force || !_readyTasks.empty() || !_unblockedTasks.empty()) {
		return CPUManager::getIdleCPU();
	} else {
		return nullptr;
	}
}


void ImmediateSuccessorScheduler::disableComputePlace(ComputePlace *computePlace)
{
	if (computePlace->_schedulerData != nullptr) {
		Task *task = (Task *) computePlace->_schedulerData;
		computePlace->_schedulerData = nullptr;
		
		std::lock_guard<SpinLock> guard(_globalLock);
		_readyTasks.push_front(task);
	}
}
