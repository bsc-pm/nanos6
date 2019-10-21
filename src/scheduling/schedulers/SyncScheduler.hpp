/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef SYNC_SCHEDULER_HPP
#define SYNC_SCHEDULER_HPP

#include <boost/lockfree/spsc_queue.hpp>

#include "MemoryAllocator.hpp"
#include "UnsyncScheduler.hpp"
#include "executors/threads/CPUManager.hpp"
#include "hardware/HardwareInfo.hpp"
#include "lowlevel/SubscriptionLock.hpp"
#include "lowlevel/TicketArraySpinLock.hpp"
#include "scheduling/SchedulerSupport.hpp"

#include <InstrumentTaskStatus.hpp>


class SyncScheduler {
protected:
	typedef boost::lockfree::spsc_queue<TaskSchedulingInfo *> add_queue_t;
	
	uint64_t _totalCPUs;
	size_t _totalAddQueues;
	size_t _totalNUMANodes;
	
	// Unsynchronized scheduler
	UnsyncScheduler *_scheduler;
	SubscriptionLock _lock;
	TicketArraySpinLock *_addQueuesLocks;
	add_queue_t *_addQueues;
	Padded<CPUNode> *_ready; // indexed by cpu_idx
	
	inline void processReadyTasks()
	{
		for (size_t i = 0; i < _totalAddQueues; i++) {
			if (!_addQueues[i].empty()) {
				_addQueues[i].consume_all(
					[&](TaskSchedulingInfo *const taskSchedulingInfo) {
						_scheduler->addReadyTask(
							taskSchedulingInfo->_task,
							taskSchedulingInfo->_computePlace,
							taskSchedulingInfo->_hint);
						delete taskSchedulingInfo;
					}
				);
			}
		}
	}
	
	inline void assignTask(uint64_t const cpuIndex, uint64_t const ticket, Task *const task)
	{
		_ready[cpuIndex].task = task;
		_ready[cpuIndex].ticket = ticket;
	}
	
	inline bool getAssignedTask(uint64_t const cpuIndex, uint64_t const myTicket, Task *&task)
	{
		task = _ready[cpuIndex].task;
		return (_ready[cpuIndex].ticket == myTicket);
	}
	
	virtual inline void setRelatedComputePlace(uint64_t, ComputePlace *)
	{
		// Do nothing
	}
	
	virtual inline ComputePlace *getRelatedComputePlace(uint64_t cpuIndex) const
	{
		const std::vector<CPU *> &cpus = CPUManager::getCPUListReference();
		return cpus[cpuIndex];
	}
	
public:
	
	SyncScheduler()
		: _lock(CPUManager::getTotalCPUs())
	{
		_totalCPUs = (uint64_t) CPUManager::getTotalCPUs();
		uint64_t totalCPUsPow2 = roundToNextPowOf2(_totalCPUs);
		assert(isPowOf2(totalCPUsPow2));
		
		_ready = (Padded<CPUNode> *) MemoryAllocator::alloc(_totalCPUs * sizeof(Padded<CPUNode>));
		for (size_t i = 0; i < _totalCPUs; i++) {
			new (&_ready[i]) Padded<CPUNode>();
		}
		
		_totalNUMANodes = HardwareInfo::getMemoryPlaceCount(nanos6_device_t::nanos6_host_device);
		
		// Using one queue per NUMA node, and a special queue for cases where there is no computePlace.
		_totalAddQueues = _totalNUMANodes + 1;
		
		_addQueues = (add_queue_t *) MemoryAllocator::alloc(_totalAddQueues * sizeof(add_queue_t));
		_addQueuesLocks = (TicketArraySpinLock *) MemoryAllocator::alloc(_totalAddQueues * sizeof(TicketArraySpinLock));
		for (size_t i = 0; i < _totalAddQueues; i++) {
			new (&_addQueues[i]) add_queue_t(totalCPUsPow2*4);
			new (&_addQueuesLocks[i]) TicketArraySpinLock(_totalCPUs);
		}
	}
	
	~SyncScheduler()
	{
		for (size_t i = 0; i < _totalAddQueues; i++) {
			_addQueues[i].~add_queue_t();
			_addQueuesLocks[i].~TicketArraySpinLock();
		}
		MemoryAllocator::free(_addQueues, _totalAddQueues * sizeof(add_queue_t));
		MemoryAllocator::free(_addQueuesLocks, _totalAddQueues * sizeof(TicketArraySpinLock));
		MemoryAllocator::free(_ready, _totalCPUs * sizeof(Padded<CPUNode>));
	}
	
	void addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint)
	{
		TaskSchedulingInfo *taskSchedulingInfo = new TaskSchedulingInfo(task, computePlace, hint);
		addTasks(computePlace, &taskSchedulingInfo, 1);
	}
	
	void addTasks(ComputePlace *computePlace, TaskSchedulingInfo *tasks[], uint64_t const size)
	{
		// If there is no CPU, use a special queue at the end that does not belong to any NUMA node.
		uint64_t const queueIndex = computePlace != nullptr ? ((CPU *)computePlace)->getNumaNodeId() : _totalAddQueues-1;
		assert(queueIndex < _totalAddQueues);
		
		if (_scheduler->priorityEnabled()) {
			for (size_t i = 0; i < size; i++) {
				// Extract task priority from taskInfo and set it as priority.
				Task *task = tasks[i]->_task;
				Task::priority_t priority = 0;
				if ((task->getTaskInfo() != nullptr) && (task->getTaskInfo()->get_priority != nullptr)) {
					task->getTaskInfo()->get_priority(task->getArgsBlock(), &priority);
					task->setPriority(priority);
					Instrument::taskHasNewPriority(task->getInstrumentationTaskId(), priority);
				}
			}
		}
		
		uint64_t count = 0;
		while (size > count) {
			// We need lock because several cpus from the same NUMA may be enqueueing at the same time.
			_addQueuesLocks[queueIndex].lock();
			count += _addQueues[queueIndex].push(tasks+count, size-count);
			_addQueuesLocks[queueIndex].unlock();
			
			if ((size > count) && _lock.tryLock()) {
				// addQueue is full, so we need to process it before pushing new tasks to it.
				processReadyTasks();
				_lock.unsubscribe();
			}
		}
	}
	
	Task *getTask(ComputePlace *computePlace, ComputePlace *deviceComputePlace);
	
	virtual Task *getReadyTask(ComputePlace *computePlace, ComputePlace *deviceComputePlace) = 0;
	
	//! \brief Check if the scheduler has available work for the current CPU
	//!
	//! \param[in] computePlace The host compute place
	inline bool hasAvailableWork(ComputePlace *computePlace)
	{
		_lock.lock();
		
		// Ensure the add queues are emptied before checking the available work
		processReadyTasks();
		
		// Check if the scheduler has work
		bool hasWork = _scheduler->hasAvailableWork(computePlace);
		
		_lock.unsubscribe();
		
		return hasWork;
	}
	
	virtual std::string getName() const = 0;
};

#endif // SYNC_SCHEDULER_HPP
