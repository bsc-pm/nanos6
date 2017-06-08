#include "PriorityScheduler.hpp"
#include "executors/threads/CPUManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "hardware/places/CPUPlace.hpp"
#include "tasks/Task.hpp"

#include <algorithm>
#include <cassert>
#include <mutex>


inline bool PriorityScheduler::TaskPriorityCompare::operator()(Task *a, Task *b)
{
	assert(a != nullptr);
	assert(b != nullptr);
	
	size_t priorityA = (size_t) a->getSchedulerInfo();
	size_t priorityB = (size_t) b->getSchedulerInfo();
	
	return (priorityA < priorityB);
}


PriorityScheduler::PriorityScheduler()
	: _pollingSlot(nullptr)
{
}

PriorityScheduler::~PriorityScheduler()
{
}


ComputePlace * PriorityScheduler::addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint, bool doGetIdle)
{
	assert(task != nullptr);
	
	size_t priority = 0;
	if ((task->getTaskInfo() != nullptr) && (task->getTaskInfo()->get_priority != nullptr)) {
		priority = task->getTaskInfo()->get_priority(task->getArgsBlock());
	}
	task->setSchedulerInfo((void *) priority);
	
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
	_readyTasks.push(task);
	
	// Attempt to get a CPU to resume the task
	if (doGetIdle) {
		return CPUManager::getIdleCPU();
	} else {
		return nullptr;
	}
}


void PriorityScheduler::taskGetsUnblocked(Task *unblockedTask, __attribute__((unused)) ComputePlace *computePlace)
{
	// 1. Attempt to send the task to a polling thread without locking
	{
		polling_slot_t *pollingSlot = _pollingSlot.load();
		while ((pollingSlot != nullptr) && !_pollingSlot.compare_exchange_strong(pollingSlot, nullptr)) {
			// Keep trying
		}
		if (pollingSlot != nullptr) {
			// Obtained the polling slot
			Task *expect = nullptr;
			
			pollingSlot->_task.compare_exchange_strong(expect, unblockedTask);
			assert(expect == nullptr);
			
			return;
		}
	}
	
	std::lock_guard<spinlock_t> guard(_globalLock);
	
	// 2. Attempt to send the task to polling thread with locking, since the polling slot
	// can only be set when locked (but unset at any time).
	{
		polling_slot_t *pollingSlot = _pollingSlot.load();
		while ((pollingSlot != nullptr) && !_pollingSlot.compare_exchange_strong(pollingSlot, nullptr)) {
			// Keep trying
		}
		if (pollingSlot != nullptr) {
			// Obtained the polling slot
			Task *expect = nullptr;
			
			pollingSlot->_task.compare_exchange_strong(expect, unblockedTask);
			assert(expect == nullptr);
			
			return;
		}
	}
	
	// 3. At this point the polling slot is empty, so send the task to the queue
	assert(_pollingSlot.load() == nullptr);
	_unblockedTasks.push(unblockedTask);
}


Task *PriorityScheduler::getReadyTask(ComputePlace *computePlace, __attribute__((unused)) Task *currentTask, bool canMarkAsIdle)
{
	std::lock_guard<spinlock_t> guard(_globalLock);
	
	size_t bestPriority = 0;
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
		bestPriority = (size_t) task->getSchedulerInfo();
		bestIs = from_immediate_successor_slot;
	}
	
	
	// 2. Check the unblocked tasks
	if (!_unblockedTasks.empty()) {
		Task *task = _unblockedTasks.top();
		assert(task != nullptr);
		
		size_t topPriority = (size_t) task->getSchedulerInfo();
		
		if ((bestIs == non_existant) || (bestPriority < topPriority)) {
			bestIs = from_unblocked_task_queue;
			bestPriority = topPriority;
		}
	}
	
	// 3. Check the ready tasks
	if (!_readyTasks.empty()) {
		Task *topTask = _readyTasks.top();
		assert(topTask != nullptr);
		
		size_t topPriority = (size_t) topTask->getSchedulerInfo();
		
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


ComputePlace *PriorityScheduler::getIdleComputePlace(bool force)
{
	std::lock_guard<spinlock_t> guard(_globalLock);
	if (force || !_readyTasks.empty() || !_unblockedTasks.empty()) {
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
		
		std::lock_guard<spinlock_t> guard(_globalLock);
		_readyTasks.push(task);
	}
}


bool PriorityScheduler::requestPolling(ComputePlace *computePlace, polling_slot_t *pollingSlot)
{
	std::lock_guard<spinlock_t> guard(_globalLock);
	
	size_t bestPriority = 0;
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
		bestPriority = (size_t) task->getSchedulerInfo();
		bestIs = from_immediate_successor_slot;
	}
	
	
	// 2. Check the unblocked tasks
	if (!_unblockedTasks.empty()) {
		Task *task = _unblockedTasks.top();
		assert(task != nullptr);
		
		size_t topPriority = (size_t) task->getSchedulerInfo();
		
		if ((bestIs == non_existant) || (bestPriority < topPriority)) {
			bestIs = from_unblocked_task_queue;
			bestPriority = topPriority;
		}
	}
	
	// 3. Check the ready tasks
	if (!_readyTasks.empty()) {
		Task *topTask = _readyTasks.top();
		assert(topTask != nullptr);
		
		size_t topPriority = (size_t) topTask->getSchedulerInfo();
		
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
		CPUManager::cpuBecomesIdle((CPU *) computePlace);
		
		return false;
	}
}


bool PriorityScheduler::releasePolling(ComputePlace *computePlace, polling_slot_t *pollingSlot)
{
	polling_slot_t *expect = pollingSlot;
	if (_pollingSlot.compare_exchange_strong(expect, nullptr)) {
		CPUManager::cpuBecomesIdle((CPU *) computePlace);
		return true;
	} else {
		return false;
	}
}


std::string PriorityScheduler::getName() const
{
	return "priority";
}

