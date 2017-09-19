/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "ImmediateSuccessorWithMultiPollingScheduler.hpp"
#include "executors/threads/CPUManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "hardware/places/CPUPlace.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"

#include <algorithm>
#include <cassert>
#include <mutex>

ImmediateSuccessorWithMultiPollingScheduler::ImmediateSuccessorWithMultiPollingScheduler(__attribute__((unused)) int numaNodeIndex)
	: _pollingSlot(nullptr)
{
}

ImmediateSuccessorWithMultiPollingScheduler::~ImmediateSuccessorWithMultiPollingScheduler()
{
}


Task *ImmediateSuccessorWithMultiPollingScheduler::getReplacementTask(__attribute__((unused)) CPU *computePlace)
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


ComputePlace * ImmediateSuccessorWithMultiPollingScheduler::addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint, bool doGetIdle)
{
	assert(task != nullptr);
	
	FatalErrorHandler::failIf(task->isTaskloop(), "Task loop not supported by this scheduler");
	
	// The following condition is only needed for the "main" task, that is added by something that is not a hardware place and thus should end up in a queue
	if (computePlace != nullptr) {
		// 1. Send the task to the immediate successor slot
		if ((hint != CHILD_TASK_HINT) && (computePlace->_schedulerData == nullptr)) {
			computePlace->_schedulerData = task;
			
			return nullptr;
		}
	}
	
	// 2. Attempt to send the task to a polling thread without locking
	{
		polling_slot_t *currentPollingSlot;
		polling_slot_t *nextPollingSlot = nullptr;
		do {
			currentPollingSlot = _pollingSlot.load();
			if (currentPollingSlot != nullptr) {
				nextPollingSlot = (polling_slot_t *) currentPollingSlot->_schedulerInfo;
			}
		} while ((currentPollingSlot != nullptr) && !_pollingSlot.compare_exchange_strong(currentPollingSlot, nextPollingSlot));
		
		if (currentPollingSlot != nullptr) {
			// Obtained the polling slot
			Task *expect = nullptr;
			
			currentPollingSlot->_task.compare_exchange_strong(expect, task);
			assert(expect == nullptr);
			
			return nullptr;
		}
	}
	
	std::lock_guard<spinlock_t> guard(_globalLock);
	
	// 3. Attempt to send the task to polling thread with locking, since the polling slot
	// can only be set when locked (but unset at any time).
	{
		polling_slot_t *currentPollingSlot;
		polling_slot_t *nextPollingSlot = nullptr;
		do {
			currentPollingSlot = _pollingSlot.load();
			if (currentPollingSlot != nullptr) {
				nextPollingSlot = (polling_slot_t *) currentPollingSlot->_atomicSchedulerInfo.load();
			}
		} while ((currentPollingSlot != nullptr) && !_pollingSlot.compare_exchange_strong(currentPollingSlot, nextPollingSlot));
		
		if (currentPollingSlot != nullptr) {
			// Obtained the polling slot
			Task *expect = nullptr;
			
			currentPollingSlot->_task.compare_exchange_strong(expect, task);
			assert(expect == nullptr);
			
			return nullptr;
		}
	}
	
	// 4. At this point the polling slot is empty, so send the task to the queue
	assert(_pollingSlot.load() == nullptr);
	_readyTasks.push_front(task);
	
	// Attempt to get a CPU to resume the task
	if (doGetIdle) {
		return CPUManager::getIdleCPU();
	} else {
		return nullptr;
	}
}


void ImmediateSuccessorWithMultiPollingScheduler::taskGetsUnblocked(Task *unblockedTask, __attribute__((unused)) ComputePlace *computePlace)
{
	// 1. Attempt to send the task to a polling thread without locking
	{
		polling_slot_t *currentPollingSlot;
		polling_slot_t *nextPollingSlot = nullptr;
		do {
			currentPollingSlot = _pollingSlot.load();
			if (currentPollingSlot != nullptr) {
				nextPollingSlot = (polling_slot_t *) currentPollingSlot->_atomicSchedulerInfo.load();
			}
		} while ((currentPollingSlot != nullptr) && !_pollingSlot.compare_exchange_strong(currentPollingSlot, nextPollingSlot));
		
		if (currentPollingSlot != nullptr) {
			// Obtained the polling slot
			Task *expect = nullptr;
			
			currentPollingSlot->_task.compare_exchange_strong(expect, unblockedTask);
			assert(expect == nullptr);
			
			return;
		}
	}
	
	std::lock_guard<spinlock_t> guard(_globalLock);
	
	// 2. Attempt to send the task to polling thread with locking, since the polling slot
	// can only be set when locked (but unset at any time).
	{
		polling_slot_t *currentPollingSlot;
		polling_slot_t *nextPollingSlot = nullptr;
		do {
			currentPollingSlot = _pollingSlot.load();
			if (currentPollingSlot != nullptr) {
				nextPollingSlot = (polling_slot_t *) currentPollingSlot->_atomicSchedulerInfo.load();
			}
		} while ((currentPollingSlot != nullptr) && !_pollingSlot.compare_exchange_strong(currentPollingSlot, nextPollingSlot));
		
		if (currentPollingSlot != nullptr) {
			// Obtained the polling slot
			Task *expect = nullptr;
			
			currentPollingSlot->_task.compare_exchange_strong(expect, unblockedTask);
			assert(expect == nullptr);
			
			return;
		}
	}
	
	// 3. At this point the polling slot is empty, so send the task to the queue
	assert(_pollingSlot.load() == nullptr);
	_unblockedTasks.push_front(unblockedTask);
}


Task *ImmediateSuccessorWithMultiPollingScheduler::getReadyTask(ComputePlace *computePlace, __attribute__((unused)) Task *currentTask, bool canMarkAsIdle)
{
	Task *task = nullptr;
	
	// 1. Get the immediate successor
	if (computePlace->_schedulerData != nullptr) {
		task = (Task *) computePlace->_schedulerData;
		computePlace->_schedulerData = nullptr;
		
		return task;
	}
	
	std::lock_guard<spinlock_t> guard(_globalLock);
	
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


ComputePlace *ImmediateSuccessorWithMultiPollingScheduler::getIdleComputePlace(bool force)
{
	std::lock_guard<spinlock_t> guard(_globalLock);
	if (force || !_readyTasks.empty() || !_unblockedTasks.empty()) {
		return CPUManager::getIdleCPU();
	} else {
		return nullptr;
	}
}


void ImmediateSuccessorWithMultiPollingScheduler::disableComputePlace(ComputePlace *computePlace)
{
	if (computePlace->_schedulerData != nullptr) {
		Task *task = (Task *) computePlace->_schedulerData;
		computePlace->_schedulerData = nullptr;
		
		std::lock_guard<spinlock_t> guard(_globalLock);
		_readyTasks.push_front(task);
	}
}


bool ImmediateSuccessorWithMultiPollingScheduler::requestPolling(ComputePlace *computePlace, polling_slot_t *pollingSlot)
{
	Task *task = nullptr;
	
	// 1. Get the immediate successor
	if (computePlace->_schedulerData != nullptr) {
		task = (Task *) computePlace->_schedulerData;
		computePlace->_schedulerData = nullptr;
		
		// Same thread, so there is no need to operate atomically
		assert(pollingSlot->_task.load() == nullptr);
		pollingSlot->_task.store(task);
		
		return true;
	}
	
	std::lock_guard<spinlock_t> guard(_globalLock);
	
	// 2. Get an unblocked task
	task = getReplacementTask((CPU *) computePlace);
	if (task != nullptr) {
		// Same thread, so there is no need to operate atomically
		assert(pollingSlot->_task.load() == nullptr);
		pollingSlot->_task.store(task);
		
		return true;
	}
	
	// 3. Or get a ready task
	if (!_readyTasks.empty()) {
		task = _readyTasks.front();
		_readyTasks.pop_front();
		
		assert(task != nullptr);
		
		// Same thread, so there is no need to operate atomically
		assert(pollingSlot->_task.load() == nullptr);
		pollingSlot->_task.store(task);
		
		return true;
	}
	
	// 4. Or attempt to get the polling slot
	polling_slot_t *currentPollingSlot;
	do {
		currentPollingSlot = _pollingSlot.load();
		pollingSlot->_atomicSchedulerInfo.store(currentPollingSlot);
	} while (!_pollingSlot.compare_exchange_strong(currentPollingSlot, pollingSlot));
	
	return true;
}


bool ImmediateSuccessorWithMultiPollingScheduler::releasePolling(ComputePlace *computePlace, polling_slot_t *pollingSlot)
{
	polling_slot_t *expect = pollingSlot;
	if (_pollingSlot.compare_exchange_strong(expect, nullptr)) {
		CPUManager::cpuBecomesIdle((CPU *) computePlace);
		return true;
	} else {
		return false;
	}
}


std::string ImmediateSuccessorWithMultiPollingScheduler::getName() const
{
	return "immediate-successor-with-multi-polling";
}
