/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef SYNC_SCHEDULER_HPP
#define SYNC_SCHEDULER_HPP

#include <atomic>

#include <boost/lockfree/spsc_queue.hpp>

#include "MemoryAllocator.hpp"
#include "UnsyncScheduler.hpp"
#include "executors/threads/CPUManager.hpp"
#include "hardware/HardwareInfo.hpp"
#include "lowlevel/DelegationLock.hpp"
#include "lowlevel/TicketArraySpinLock.hpp"
#include "scheduling/SchedulerSupport.hpp"


class SyncScheduler {
protected:
	//! Scheduler's device type
	nanos6_device_t _deviceType;

	//! Underlying unsynchronized scheduler
	UnsyncScheduler *_scheduler;

private:
	typedef boost::lockfree::spsc_queue<Task *, boost::lockfree::allocator<TemplateAllocator<Task *>>> add_queue_t;

	//! Total number of computePlaces
	uint64_t _totalComputePlaces;

	//! Total number of add queues
	size_t _totalAddQueues;

	//! Delegation lock protecting the access
	//! to the unsychronized scheduler
	DelegationLock<Task *> _lock;

	//! Locks for adding tasks to the add queues
	TicketArraySpinLock *_addQueuesLocks;

	//! Add queues of ready tasks
	add_queue_t *_addQueues;

	//! Indicates whether there is any compute place
	//! serving tasks inside the scheduling loop
	std::atomic<bool> _servingTasks;

	//! The limit of tasks that a compute place can serve within a
	//! single burst in the scheduling loop. This avoids that an
	//! external compute place gets stuck solely serving tasks for
	//! too much time
	size_t _maxServedTasks;

public:
	//! NOTE We initialize the delegation lock with 2 * numCPUs since some
	//! threads may oversubscribe and thus we may need more than numCPUs
	//! slots in the lock's waiting queue
	SyncScheduler(size_t totalComputePlaces, nanos6_device_t deviceType = nanos6_host_device) :
		_deviceType(deviceType),
		_scheduler(nullptr),
		_totalComputePlaces(totalComputePlaces),
		_lock((uint64_t) totalComputePlaces * 2),
		_servingTasks(false),
		_maxServedTasks(totalComputePlaces * 20)
	{
		uint64_t totalCPUsPow2 = SchedulerSupport::roundToNextPowOf2(_totalComputePlaces);
		assert(SchedulerSupport::isPowOf2(totalCPUsPow2));

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

		delete _scheduler;
	}

	virtual inline nanos6_device_t getDeviceType()
	{
		return _deviceType;
	}

	//! \brief Check whether a compute place is serving tasks
	inline bool isServingTasks()
	{
		return _servingTasks.load(std::memory_order_relaxed);
	}

	inline void addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint)
	{
		// TODO: Allow adding multiple tasks in the future
		addReadyTasks(&task, 1, computePlace, hint);
	}

	inline void addReadyTasks(Task *tasks[], const size_t numTasks, ComputePlace *computePlace, ReadyTaskHint hint)
	{
		// Use a special queue not belonging to any NUMA node if no compute place
		const size_t queueIndex = (computePlace != nullptr) ? ((CPU *)computePlace)->getNumaNodeId() : _totalAddQueues-1;
		assert(queueIndex < _totalAddQueues);

		for (size_t t = 0; t < numTasks; t++) {
			assert(tasks[t] != nullptr);
			// Set temporary info that is used when processing ready tasks
			tasks[t]->setComputePlace(computePlace);
			tasks[t]->setSchedulingHint(hint);
			tasks[t]->computeTaskAffinity();
			tasks[t]->computeNUMAAffinity();
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
				_lock.unlock();
			}
		}
	}

	Task *getTask(ComputePlace *computePlace);

	virtual Task *getReadyTask(ComputePlace *computePlace) = 0;

	virtual std::string getName() const = 0;

private:
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

	//! \brief Set serving tasks condition
	//!
	//! This function should be called only after acquiring the scheduler
	//! lock and before releasing it, which is the moment when the compute
	//! place is responsible for serving tasks
	//!
	//! \param[in] servingTasks The value to be set
	inline void setServingTasks(bool servingTasks)
	{
		_servingTasks.store(servingTasks, std::memory_order_relaxed);
	}

	//! \brief Get the compute place by its index
	//!
	//! \param[in] computePlaceIndex The index of the compute place
	//!
	//! \return The corresponding compute place
	virtual ComputePlace *getComputePlace(uint64_t computePlaceIndex) const = 0;

	//! \brief Check whether a compute place should stop serving tasks
	//!
	//! This function should be called from the compute place that is
	//! serving tasks, which is the one that acquired the subscription
	//! lock and is serving ready tasks to the rest of compute places
	//!
	//! \param[in] computePlace The compute place that is serving tasks
	//!
	//! \return Whether the compute place should stop
	virtual bool mustStopServingTasks(ComputePlace *computePlace) const = 0;

	//! \brief Perform the required actions after serving tasks
	//!
	//! This function is called when a compute place has stopped
	//! serving tasks, which is the one that served ready tasks
	//! to the rest of compute places
	//!
	//! \param[in] computePlace The compute place that was serving tasks
	//! \param[in] assignedTask The task that has served to itself (if any)
	virtual void postServingTasks(ComputePlace *computePlace, Task *assignedTask) = 0;
};

#endif // SYNC_SCHEDULER_HPP
