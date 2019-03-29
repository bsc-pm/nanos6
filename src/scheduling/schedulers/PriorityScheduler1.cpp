/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "PriorityScheduler1.hpp"
#include "executors/threads/CPUManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "hardware/places/CPUPlace.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"

#include <InstrumentTaskStatus.hpp>

#include <algorithm>
#include <cassert>
#include <mutex>


inline bool PriorityScheduler1::TaskPriorityCompare::operator()(Task *a, Task *b)
{
	assert(a != nullptr);
	assert(b != nullptr);
	
	return (a->getPriority() < b->getPriority());
}


PriorityScheduler1::PriorityScheduler1(__attribute__((unused)) int numaNodeIndex)
	: _pollingSlot(nullptr)
{
}

PriorityScheduler1::~PriorityScheduler1()
{
}


ComputePlace * PriorityScheduler1::addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint, bool doGetIdle)
{
	assert(task != nullptr);
	
	FatalErrorHandler::failIf(task->getDeviceType() != nanos6_device_t::nanos6_host_device, "Device tasks not supported by this scheduler");	
	FatalErrorHandler::failIf(task->isTaskloop(), "Task loop not supported by this scheduler");
	
	Task::priority_t priority = 0;
	if ((task->getTaskInfo() != nullptr) && (task->getTaskInfo()->get_priority != nullptr)) {
		task->getTaskInfo()->get_priority(task->getArgsBlock(), &priority);
		task->setPriority(priority);
		Instrument::taskHasNewPriority(task->getInstrumentationTaskId(), priority);
	}
	
	// The following condition is only needed for the "main" task, that is added by something that is not a hardware place and thus should end up in a queue
	if (computePlace != nullptr) {
		// 1. Send the task to the immediate successor slot
		if ((hint != CHILD_TASK_HINT) && (hint != UNBLOCKED_TASK_HINT) && (hint != BUSY_COMPUTE_PLACE_TASK_HINT) && (computePlace->_schedulerData == nullptr)) {
			computePlace->_schedulerData = task;
			
			return nullptr;
		}
	}
	
	// 2. Attempt to send the task to a polling thread without locking
	{
		polling_slot_t *pollingSlot = _pollingSlot.load();
		while ((pollingSlot != nullptr) && !_pollingSlot.compare_exchange_strong(pollingSlot, nullptr)) {
			// Keep trying
		}
		if (pollingSlot != nullptr) {
			// Obtained the polling slot
			Task *expect = nullptr;
			
			pollingSlot->_task.compare_exchange_strong(expect, task);
			assert(expect == nullptr);
			
			return nullptr;
		}
	}
	
	std::lock_guard<spinlock_t> guard(_globalLock);
	
	// 3. Attempt to send the task to polling thread with locking, since the polling slot
	// can only be set when locked (but unset at any time).
	{
		polling_slot_t *pollingSlot = _pollingSlot.load();
		while ((pollingSlot != nullptr) && !_pollingSlot.compare_exchange_strong(pollingSlot, nullptr)) {
			// Keep trying
		}
		if (pollingSlot != nullptr) {
			// Obtained the polling slot
			Task *expect = nullptr;
			
			pollingSlot->_task.compare_exchange_strong(expect, task);
			assert(expect == nullptr);
			
			return nullptr;
		}
	}
	
	// 4. At this point the polling slot is empty, so send the task to the queue
	assert(_pollingSlot.load() == nullptr);
	if (hint == UNBLOCKED_TASK_HINT) {
		_unblockedTasks.push(task);
	} else {
		_readyTasks.push(task);
	}
	
	
	// Attempt to get a CPU to resume the task
	if (doGetIdle) {
		return CPUManager::getIdleCPU();
	} else {
		return nullptr;
	}
}


Task *PriorityScheduler1::getReadyTask(ComputePlace *computePlace, __attribute__((unused)) Task *currentTask, bool canMarkAsIdle, __attribute__((unused)) bool doWait)
{
	if (computePlace->getType() != nanos6_device_t::nanos6_host_device) {
		return nullptr;
	}
	
	std::lock_guard<spinlock_t> guard(_globalLock);
	
	Task::priority_t bestPriority = 0;
	enum {
		non_existant = 0,
		from_immediate_successor_slot,
		from_unblocked_task_queue,
		from_ready_task_queue
	} bestIs = non_existant;
	
	// 1. Check the immediate successor
	bool haveImmediateSuccessor = (computePlace->_schedulerData != nullptr);
	if (haveImmediateSuccessor) {
		Task *task = (Task *) computePlace->_schedulerData;
		bestPriority = task->getPriority();
		bestIs = from_immediate_successor_slot;
	}
	
	
	// 2. Check the unblocked tasks
	if (!_unblockedTasks.empty()) {
		Task *task = _unblockedTasks.top();
		assert(task != nullptr);
		
		Task::priority_t topPriority = task->getPriority();
		
		if ((bestIs == non_existant) || (bestPriority < topPriority)) {
			bestIs = from_unblocked_task_queue;
			bestPriority = topPriority;
		}
	}
	
	// 3. Check the ready tasks
	if (!_readyTasks.empty()) {
		Task *topTask = _readyTasks.top();
		assert(topTask != nullptr);
		
		Task::priority_t topPriority = topTask->getPriority();
		
		if ((bestIs == non_existant) || (bestPriority < topPriority)) {
			bestIs = from_ready_task_queue;
			bestPriority = topPriority;
		}
	}
	
	// 4. Queue the immediate successor if necessary and return the choosen task
	if (bestIs != non_existant) {
		// The immediate successor was choosen
		if (bestIs == from_immediate_successor_slot) {
			Task *task = (Task *) computePlace->_schedulerData;
			computePlace->_schedulerData = nullptr;
			
			return task;
		}
		
		// After this point the immediate successor was not choosen
		
		// Queue the immediate successor
		if (haveImmediateSuccessor) {
			assert(bestIs != from_immediate_successor_slot);
			
			Task *task = (Task *) computePlace->_schedulerData;
			computePlace->_schedulerData = nullptr;
			
			_readyTasks.push(task);
		}
		
		if (bestIs == from_ready_task_queue) {
			Task *task = _readyTasks.top();
			_readyTasks.pop();
			assert(task != nullptr);
			
			return task;
		}
		
		if (bestIs == from_unblocked_task_queue) {
			Task *task = _unblockedTasks.top();
			_unblockedTasks.pop();
			assert(task != nullptr);
			
			return task;
		}
		
		assert("Internal logic error" == nullptr);
	}
	
	assert(bestIs == non_existant);
	
	// 4. Or mark the CPU as idle
	if (canMarkAsIdle) {
		CPUManager::cpuBecomesIdle((CPU *) computePlace);
	}
	
	return nullptr;
}


ComputePlace *PriorityScheduler1::getIdleComputePlace(bool force)
{
	std::lock_guard<spinlock_t> guard(_globalLock);
	if (force || !_readyTasks.empty() || !_unblockedTasks.empty()) {
		return CPUManager::getIdleCPU();
	} else {
		return nullptr;
	}
}


void PriorityScheduler1::disableComputePlace(ComputePlace *computePlace)
{
	if (computePlace->_schedulerData != nullptr) {
		Task *task = (Task *) computePlace->_schedulerData;
		computePlace->_schedulerData = nullptr;
		
		std::lock_guard<spinlock_t> guard(_globalLock);
		_readyTasks.push(task);
	}
}


bool PriorityScheduler1::requestPolling(ComputePlace *computePlace, polling_slot_t *pollingSlot, bool canMarkAsIdle)
{
	std::lock_guard<spinlock_t> guard(_globalLock);
	
	Task::priority_t bestPriority = 0;
	enum {
		non_existant = 0,
		from_immediate_successor_slot,
		from_unblocked_task_queue,
		from_ready_task_queue
	} bestIs = non_existant;
	
	// 1. Check the immediate successor
	bool haveImmediateSuccessor = (computePlace->_schedulerData != nullptr);
	if (haveImmediateSuccessor) {
		Task *task = (Task *) computePlace->_schedulerData;
		bestPriority = task->getPriority();
		bestIs = from_immediate_successor_slot;
	}
	
	
	// 2. Check the unblocked tasks
	if (!_unblockedTasks.empty()) {
		Task *task = _unblockedTasks.top();
		assert(task != nullptr);
		
		Task::priority_t topPriority = task->getPriority();
		
		if ((bestIs == non_existant) || (bestPriority < topPriority)) {
			bestIs = from_unblocked_task_queue;
			bestPriority = topPriority;
		}
	}
	
	// 3. Check the ready tasks
	if (!_readyTasks.empty()) {
		Task *topTask = _readyTasks.top();
		assert(topTask != nullptr);
		
		Task::priority_t topPriority = topTask->getPriority();
		
		if ((bestIs == non_existant) || (bestPriority < topPriority)) {
			bestIs = from_ready_task_queue;
			bestPriority = topPriority;
		}
	}
	
	// 4. Queue the immediate successor if necessary and return the choosen task
	if (bestIs != non_existant) {
		// The immediate successor was choosen
		if (bestIs == from_immediate_successor_slot) {
			Task *task = (Task *) computePlace->_schedulerData;
			computePlace->_schedulerData = nullptr;
			
			// Same thread, so there is no need to operate atomically
			assert(pollingSlot->_task.load() == nullptr);
			pollingSlot->_task.store(task);
			
			return true;
		}
		
		// After this point the immediate successor was not choosen
		
		// Queue the immediate successor
		if (haveImmediateSuccessor) {
			assert(bestIs != from_immediate_successor_slot);
			
			Task *task = (Task *) computePlace->_schedulerData;
			computePlace->_schedulerData = nullptr;
			
			_readyTasks.push(task);
		}
		
		if (bestIs == from_ready_task_queue) {
			Task *task = _readyTasks.top();
			_readyTasks.pop();
			assert(task != nullptr);
			
			// Same thread, so there is no need to operate atomically
			assert(pollingSlot->_task.load() == nullptr);
			pollingSlot->_task.store(task);
			
			return true;
		}
		
		if (bestIs == from_unblocked_task_queue) {
			Task *task = _unblockedTasks.top();
			_unblockedTasks.pop();
			assert(task != nullptr);
			
			// Same thread, so there is no need to operate atomically
			assert(pollingSlot->_task.load() == nullptr);
			pollingSlot->_task.store(task);
			
			return true;
		}
		
		assert("Internal logic error" == nullptr);
	}
	
	assert(bestIs == non_existant);
	
	// 4. Or attempt to get the polling slot
	polling_slot_t *expect = nullptr;
	if (_pollingSlot.compare_exchange_strong(expect, pollingSlot)) {
		
		// 4.a. Successful
		return true;
	} else {
		// 5.b. There is already another thread polling. Therefore, mark the CPU as idle
		if (canMarkAsIdle) {
			CPUManager::cpuBecomesIdle((CPU *) computePlace);
		}
		
		return false;
	}
}


bool PriorityScheduler1::releasePolling(ComputePlace *computePlace, polling_slot_t *pollingSlot, bool canMarkAsIdle)
{
	polling_slot_t *expect = pollingSlot;
	if (_pollingSlot.compare_exchange_strong(expect, nullptr)) {
		if (canMarkAsIdle) {
			CPUManager::cpuBecomesIdle((CPU *) computePlace);
		}
		return true;
	} else {
		return false;
	}
}


std::string PriorityScheduler1::getName() const
{
	return "priority";
}

