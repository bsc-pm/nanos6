/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#include "NoSleepPriorityScheduler.hpp"
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


inline bool NoSleepPriorityScheduler::TaskPriorityCompare::operator()(Task *a, Task *b)
{
	assert(a != nullptr);
	assert(b != nullptr);
	
	return (a->getPriority() < b->getPriority());
}


NoSleepPriorityScheduler::NoSleepPriorityScheduler(__attribute__((unused)) int numaNodeIndex)
{
}

NoSleepPriorityScheduler::~NoSleepPriorityScheduler()
{
}


ComputePlace * NoSleepPriorityScheduler::addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint, bool doGetIdle)
{
	assert(task != nullptr);
	
	FatalErrorHandler::failIf(task->getDeviceType() != nanos6_device_t::nanos6_host_device, "Device tasks not supported by this scheduler");
	
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
	
	std::lock_guard<spinlock_t> guard(_globalLock);
	
	// 2. Attempt to send the task to a polling thread
	if (!_pollingSlots.empty()) {
		polling_slot_t *pollingSlot = _pollingSlots.front();
		_pollingSlots.pop_front();
		
		assert(pollingSlot != nullptr);
		pollingSlot->_task.store(task);
		
		return nullptr;
	}
	
	// 3. At this point the polling slot queue is empty, so send the task to the queue
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


Task *NoSleepPriorityScheduler::getReadyTask(ComputePlace *computePlace, __attribute__((unused)) Task *currentTask, bool canMarkAsIdle, __attribute__((unused)) bool doWait)
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
	
	// 4. Queue the immediate successor if necessary and return the chosen task
	if (bestIs != non_existant) {
		// The immediate successor was chosen
		if (bestIs == from_immediate_successor_slot) {
			Task *task = (Task *) computePlace->_schedulerData;
			computePlace->_schedulerData = nullptr;
			
			return task;
		}
		
		// After this point the immediate successor was not chosen
		
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


ComputePlace *NoSleepPriorityScheduler::getIdleComputePlace(bool force)
{
	std::lock_guard<spinlock_t> guard(_globalLock);
	if (force || !_readyTasks.empty() || !_unblockedTasks.empty()) {
		return CPUManager::getIdleCPU();
	} else {
		return nullptr;
	}
}


void NoSleepPriorityScheduler::disableComputePlace(ComputePlace *computePlace)
{
	if (computePlace->_schedulerData != nullptr) {
		Task *task = (Task *) computePlace->_schedulerData;
		computePlace->_schedulerData = nullptr;
		
		std::lock_guard<spinlock_t> guard(_globalLock);
		_readyTasks.push(task);
	}
}


bool NoSleepPriorityScheduler::requestPolling(ComputePlace *computePlace, polling_slot_t *pollingSlot, __attribute__((unused)) bool canMarkAsIdle)
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
	
	// 4. Queue the immediate successor if necessary and return the chosen task
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
		
		// After this point the immediate successor was not chosen
		
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
	
	// 5. Or queue the polling slot
	_pollingSlots.push_back(pollingSlot);
	return true;
}


bool NoSleepPriorityScheduler::releasePolling(ComputePlace *computePlace, polling_slot_t *pollingSlot, bool canMarkAsIdle)
{
	std::lock_guard<spinlock_t> guard(_globalLock);
	
	auto it = std::find(_pollingSlots.begin(), _pollingSlots.end(), pollingSlot);
	assert((it != _pollingSlots.end()) || (pollingSlot->_task.load() != nullptr));
	
	if (it == _pollingSlots.end()) {
		// Too late, a task has already been assigned
		return false;
	} else {
		_pollingSlots.erase(it);
		
		if (canMarkAsIdle) {
			CPUManager::cpuBecomesIdle((CPU *) computePlace);
		}
		return true;
	}
}


std::string NoSleepPriorityScheduler::getName() const
{
	return "nosleep-priority";
}

