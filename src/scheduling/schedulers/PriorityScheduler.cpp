/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "PriorityScheduler.hpp"
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


PriorityScheduler::PriorityScheduler(__attribute__((unused)) int numaNodeIndex)
	: _pollingSlot(nullptr)
{
}

PriorityScheduler::~PriorityScheduler()
{
}


ComputePlace * PriorityScheduler::addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint, bool doGetIdle)
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
	
	_globalLock.lock();
	
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
			
			_globalLock.unlock();
			return nullptr;
		}
	}
	
	// 4. At this point the polling slot is empty, so send the task to the queue
	assert(_pollingSlot.load() == nullptr);
	PriorityClass &priorityClass = _tasks[-priority];
	
	priorityClass._lock.lock();
	_globalLock.unlock();
	
	if ((hint == UNBLOCKED_TASK_HINT) || (hint == CHILD_TASK_HINT)) {
		priorityClass._queue.push_front(task);
	} else {
		priorityClass._queue.push_back(task);
	}
	priorityClass._lock.unlock();
	
	// Attempt to get a CPU to resume the task
	if (doGetIdle) {
		return CPUManager::getIdleCPU();
	} else {
		return nullptr;
	}
}


Task *PriorityScheduler::getReadyTask(ComputePlace *computePlace, __attribute__((unused)) Task *currentTask, bool canMarkAsIdle, __attribute__((unused)) bool doWait)
{
	if (computePlace->getType() != nanos6_device_t::nanos6_host_device) {
		return nullptr;
	}
	
	Task::priority_t bestPriority = 0;
	enum {
		non_existant = 0,
		from_immediate_successor_slot,
		from_task_queue
	} bestIs = non_existant;
	
	Task *immediateSuccessor = nullptr;
	Task *queuedTask = nullptr;
	
	// 1. Check the immediate successor
	immediateSuccessor = (Task *) computePlace->_schedulerData;
	if (immediateSuccessor != nullptr) {
		bestPriority = immediateSuccessor->getPriority();
		bestIs = from_immediate_successor_slot;
	}
	
	_globalLock.lock();
	
	// 2. Check the highest priority class
	std::map</* Task::priority_t */ long, PriorityClass>::iterator it;
	PriorityClass *priorityClass = nullptr;
	if (!_tasks.empty()) {
		it = _tasks.begin();
		assert(it != _tasks.end());
		
		priorityClass = &it->second;
		priorityClass->_lock.lock();
		
		assert(!priorityClass->_queue.empty());
		queuedTask = priorityClass->_queue.front();
		assert(queuedTask != nullptr);
		
		Task::priority_t topPriority = queuedTask->getPriority();
		
		if ((bestIs == non_existant) || (bestPriority < topPriority)) {
			bestIs = from_task_queue;
			bestPriority = topPriority;
		} else {
			priorityClass->_lock.unlock();
			priorityClass = nullptr;
		}
	}
	
	// 3. Queue the immediate successor if necessary and return the chosen task
	if (bestIs != non_existant) {
		// The immediate successor was chosen
		if (bestIs == from_immediate_successor_slot) {
			_globalLock.unlock();
			computePlace->_schedulerData = nullptr;
			
			return immediateSuccessor;
		}
		
		// After this point the immediate successor was not chosen
		assert(bestIs == from_task_queue);
		assert(queuedTask != nullptr);
		priorityClass->_queue.pop_front();
		if (priorityClass->_queue.empty()) {
			priorityClass->_lock.unlock();
			priorityClass = nullptr;
			_tasks.erase(it);
		}
		
		// Queue the immediate successor
		if (immediateSuccessor != nullptr) {
			assert(bestIs != from_immediate_successor_slot);
			
			// Clear the immediate successor
			computePlace->_schedulerData = nullptr;
			
			Task::priority_t immediateSuccessorPriority = immediateSuccessor->getPriority();
			PriorityClass *immediateSuccessorPriorityClass = &_tasks[-immediateSuccessorPriority];
			
			if (immediateSuccessorPriorityClass == priorityClass) {
				priorityClass->_queue.push_back(immediateSuccessor);
				priorityClass->_lock.unlock();
			} else {
				if (priorityClass != nullptr) {
					priorityClass->_lock.unlock();
				}
				immediateSuccessorPriorityClass->_lock.lock();
				immediateSuccessorPriorityClass->_queue.push_back(immediateSuccessor);
				immediateSuccessorPriorityClass->_lock.unlock();
			}
		} else if (priorityClass != nullptr) {
			priorityClass->_lock.unlock();
		}
		
		_globalLock.unlock();
		return queuedTask;
	}
	assert(bestIs == non_existant);
	
	// 4. Or mark the CPU as idle
	if (canMarkAsIdle) {
		CPUManager::cpuBecomesIdle((CPU *) computePlace);
	}
	_globalLock.unlock();
	
	return nullptr;
}


ComputePlace *PriorityScheduler::getIdleComputePlace(bool force)
{
	std::lock_guard<spinlock_t> guard(_globalLock);
	if (force || !_tasks.empty()) {
		return CPUManager::getIdleCPU();
	} else {
		return nullptr;
	}
}


void PriorityScheduler::disableComputePlace(ComputePlace *computePlace)
{
	if (computePlace->_schedulerData != nullptr) {
		Task *task = (Task *) computePlace->_schedulerData;
		computePlace->_schedulerData = nullptr;
		
		Task::priority_t priority = task->getPriority();
		
		_globalLock.lock();
		PriorityClass &priorityClass = _tasks[-priority];
		
		priorityClass._lock.lock();
		_globalLock.unlock();
		
		priorityClass._queue.push_back(task);
		priorityClass._lock.unlock();
	}
}


bool PriorityScheduler::requestPolling(ComputePlace *computePlace, polling_slot_t *pollingSlot, bool canMarkAsIdle)
{
	Task::priority_t bestPriority = 0;
	enum {
		non_existant = 0,
		from_immediate_successor_slot,
		from_task_queue
	} bestIs = non_existant;
	
	Task *immediateSuccessor = nullptr;
	Task *queuedTask = nullptr;
	
	// 1. Check the immediate successor
	immediateSuccessor = (Task *) computePlace->_schedulerData;
	if (immediateSuccessor != nullptr) {
		bestPriority = immediateSuccessor->getPriority();
		bestIs = from_immediate_successor_slot;
	}
	
	_globalLock.lock();
	
	// 2. Check the highest priority class
	std::map</* Task::priority_t */ long, PriorityClass>::iterator it;
	PriorityClass *priorityClass = nullptr;
	if (!_tasks.empty()) {
		it = _tasks.begin();
		assert(it != _tasks.end());
		
		priorityClass = &it->second;
		priorityClass->_lock.lock();
		
		assert(!priorityClass->_queue.empty());
		queuedTask = priorityClass->_queue.front();
		assert(queuedTask != nullptr);
		
		Task::priority_t topPriority = queuedTask->getPriority();
		
		if ((bestIs == non_existant) || (bestPriority < topPriority)) {
			bestIs = from_task_queue;
			bestPriority = topPriority;
		} else {
			priorityClass->_lock.unlock();
			priorityClass = nullptr;
		}
	}
	
	// 3. Queue the immediate successor if necessary and return the chosen task
	if (bestIs != non_existant) {
		// The immediate successor was chosen
		if (bestIs == from_immediate_successor_slot) {
			_globalLock.unlock();
			
			computePlace->_schedulerData = nullptr;
			
			// Same thread, so there is no need to operate atomically
			assert(pollingSlot->_task.load() == nullptr);
			pollingSlot->_task.store(immediateSuccessor);
			
			return true;
		}
		
		// After this point the immediate successor was not chosen
		assert(bestIs == from_task_queue);
		assert(queuedTask != nullptr);
		priorityClass->_queue.pop_front();
		if (priorityClass->_queue.empty()) {
			priorityClass->_lock.unlock();
			priorityClass = nullptr;
			_tasks.erase(it);
		}
		
		// Queue the immediate successor
		if (immediateSuccessor != nullptr) {
			assert(bestIs != from_immediate_successor_slot);
			
			// Clear the immediate successor
			computePlace->_schedulerData = nullptr;
			
			Task::priority_t immediateSuccessorPriority = immediateSuccessor->getPriority();
			PriorityClass *immediateSuccessorPriorityClass = &_tasks[-immediateSuccessorPriority];
			
			if (immediateSuccessorPriorityClass == priorityClass) {
				priorityClass->_queue.push_back(immediateSuccessor);
				priorityClass->_lock.unlock();
			} else {
				if (priorityClass != nullptr) {
					priorityClass->_lock.unlock();
				}
				immediateSuccessorPriorityClass->_lock.lock();
				immediateSuccessorPriorityClass->_queue.push_back(immediateSuccessor);
				immediateSuccessorPriorityClass->_lock.unlock();
			}
		} else if (priorityClass != nullptr) {
			priorityClass->_lock.unlock();
		}
		
		_globalLock.unlock();
		
		// Same thread, so there is no need to operate atomically
		assert(pollingSlot->_task.load() == nullptr);
		pollingSlot->_task.store(queuedTask);
		
		return true;
	}
	
	assert(bestIs == non_existant);
	
	// 4. Or attempt to get the polling slot
	polling_slot_t *expect = nullptr;
	if (_pollingSlot.compare_exchange_strong(expect, pollingSlot)) {
		
		// 4.a. Successful
		_globalLock.unlock();
		return true;
	} else {
		// 5.b. There is already another thread polling. Therefore, mark the CPU as idle
		if (canMarkAsIdle) {
			CPUManager::cpuBecomesIdle((CPU *) computePlace);
		}
		_globalLock.unlock();
		
		return false;
	}
}


bool PriorityScheduler::releasePolling(ComputePlace *computePlace, polling_slot_t *pollingSlot, bool canMarkAsIdle)
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


std::string PriorityScheduler::getName() const
{
	return "priority";
}

