/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
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


class SyncScheduler {
protected:
	typedef boost::lockfree::spsc_queue<Task *> add_queue_t;
	typedef Padded<SchedulerSupport::CPUNode> PaddedCPUNode;

	/* Members */
	nanos6_device_t _deviceType;
	uint64_t _totalComputePlaces;
	size_t _totalAddQueues;

	// Unsynchronized scheduler
	UnsyncScheduler *_scheduler;
	SubscriptionLock _lock;
	TicketArraySpinLock *_addQueuesLocks;
	add_queue_t *_addQueues;
	PaddedCPUNode *_ready; // indexed by CPU index

	//! \brief Transfer ready tasks from lock-free queues to the scheduler
	//!
	//! This function moves all ready tasks that have been added to
	//! the lock-free queues (one per NUMA) to the definitive ready
	//! task queue. Note this function must be called with the lock
	//! of the scheduler acquired
	inline void processReadyTasks()
	{
		for (size_t i = 0; i < _totalAddQueues; i++) {
			if (!_addQueues[i].empty()) {
				_addQueues[i].consume_all(
					[&](Task *task) {
						// Add the task to the unsync scheduler
						_scheduler->addReadyTask(task,
							task->getComputePlace(),
							task->getSchedulingHint());
						// Reset compute place for security
						task->setComputePlace(nullptr);
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

	static inline ComputePlace *getComputePlace(nanos6_device_t deviceType, uint64_t computePlaceIndex)
	{
		if (deviceType == nanos6_host_device) {
			const std::vector<CPU *> &cpus = CPUManager::getCPUListReference();
			return cpus[computePlaceIndex];
		} else {
			return HardwareInfo::getComputePlace(deviceType, computePlaceIndex);
		}
	}

public:

	//! NOTE We initialize the SubscriptionLock with 2 * numCPUs since some
	//! threads may oversubscribe and thus we may need more than "numCPUs"
	//! slots in the lock's waiting queue
	SyncScheduler(size_t totalComputePlaces, nanos6_device_t deviceType = nanos6_host_device) :
		_deviceType(deviceType),
		_totalComputePlaces(totalComputePlaces),
		_lock((uint64_t) totalComputePlaces * 2)
	{
		uint64_t totalCPUsPow2 = SchedulerSupport::roundToNextPowOf2(_totalComputePlaces);
		assert(SchedulerSupport::isPowOf2(totalCPUsPow2));

		_ready = (PaddedCPUNode *) MemoryAllocator::alloc(_totalComputePlaces * sizeof(PaddedCPUNode));

		for (size_t i = 0; i < _totalComputePlaces; i++) {
			new (&_ready[i]) PaddedCPUNode();
		}

		size_t totalNUMANodes = HardwareInfo::getMemoryPlaceCount(nanos6_host_device);

		// Use a queue per NUMA node and a special queue
		// for cases where there is no compute place
		_totalAddQueues = totalNUMANodes + 1;

		_addQueues = (add_queue_t *)
			MemoryAllocator::alloc(_totalAddQueues * sizeof(add_queue_t));
		_addQueuesLocks = (TicketArraySpinLock *)
			MemoryAllocator::alloc(_totalAddQueues * sizeof(TicketArraySpinLock));

		for (size_t i = 0; i < _totalAddQueues; i++) {
			new (&_addQueues[i]) add_queue_t(totalCPUsPow2*4);
			new (&_addQueuesLocks[i]) TicketArraySpinLock(_totalComputePlaces);
		}
	}

	virtual ~SyncScheduler()
	{
		for (size_t i = 0; i < _totalAddQueues; i++) {
			_addQueues[i].~add_queue_t();
			_addQueuesLocks[i].~TicketArraySpinLock();
		}
		MemoryAllocator::free(_addQueues, _totalAddQueues * sizeof(add_queue_t));
		MemoryAllocator::free(_addQueuesLocks, _totalAddQueues * sizeof(TicketArraySpinLock));
		MemoryAllocator::free(_ready, _totalComputePlaces * sizeof(PaddedCPUNode));

		delete _scheduler;
	}

	virtual nanos6_device_t getDeviceType()
	{
		return _deviceType;
	}

	void addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint)
	{
		// TODO: Allow adding multiple tasks in the future
		addTasks(&task, 1, computePlace, hint);
	}

	void addTasks(Task *tasks[], const size_t numTasks, ComputePlace *computePlace, ReadyTaskHint hint)
	{
		// Use a special queue not belonging to any NUMA node if no compute place
		const size_t queueIndex = (computePlace != nullptr) ? ((CPU *)computePlace)->getNumaNodeId() : _totalAddQueues-1;
		assert(queueIndex < _totalAddQueues);

		for (size_t t = 0; t < numTasks; t++) {
			assert(tasks[t] != nullptr);
			// Set temporary info that is used when processing ready tasks
			tasks[t]->setComputePlace(computePlace);
			tasks[t]->setSchedulingHint(hint);
		}

		size_t count = 0;
		while (numTasks > count) {
			// Acquire lock since other cpus from the same NUMA may be enqueueing
			_addQueuesLocks[queueIndex].lock();
			count += _addQueues[queueIndex].push(tasks+count, numTasks-count);
			_addQueuesLocks[queueIndex].unlock();

			if ((numTasks > count) && _lock.tryLock()) {
				// Process queues before pushing new tasks
				processReadyTasks();
				_lock.unsubscribe();
			}
		}
	}

	Task *getTask(ComputePlace *computePlace);

	virtual Task *getReadyTask(ComputePlace *computePlace) = 0;

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

	//! \brief Notify the scheduler that a CPU is about to be disabled
	//! in case any tasks must be unassigned
	//!
	//! \param[in] cpuId The id of the cpu that will be disabled
	//! \param[in] task A task assigned to the current thread or nullptr
	//!
	//! \return Whether work was reassigned upon disabling this CPU
	inline bool disablingCPU(size_t cpuId, Task * task)
	{
		_lock.lock();

		// Ensure the add queues are emptied
		processReadyTasks();
		bool tasksReassigned = _scheduler->disablingCPU(cpuId, task);

		_lock.unsubscribe();

		return tasksReassigned;
	}

	virtual std::string getName() const = 0;
};

#endif // SYNC_SCHEDULER_HPP
