/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef HOST_UNSYNC_SCHEDULER_HPP
#define HOST_UNSYNC_SCHEDULER_HPP

#include "UnsyncScheduler.hpp"
#include "scheduling/ready-queues/DeadlineQueue.hpp"
#include "scheduling/ready-queues/ReadyQueueDeque.hpp"
#include "scheduling/ready-queues/ReadyQueueMap.hpp"

class HostUnsyncScheduler : public UnsyncScheduler {
public:
	HostUnsyncScheduler(SchedulingPolicy policy, bool enablePriority) :
		UnsyncScheduler(policy, enablePriority)
	{
		_deadlineTasks = new DeadlineQueue(policy);
		assert(_deadlineTasks != nullptr);

		_numQueues = NUMAManager::getTrackingNodes();
		assert(_numQueues > 0);

		_queues = (ReadyQueue **) MemoryAllocator::alloc(_numQueues * sizeof(ReadyQueue *));
		assert(_queues != nullptr);

		for (uint64_t i = 0; i < _numQueues; i++) {
			if (NUMAManager::isValidNUMA(i) || _numQueues == 1) {
				// In case there is a single queue we have to create it always
				// in the first queue position
				if (enablePriority) {
					_queues[i] = new ReadyQueueMap(policy);
				} else {
					_queues[i] = new ReadyQueueDeque(policy);
				}
			} else {
				_queues[i] = nullptr;
			}
		}
	}

	virtual ~HostUnsyncScheduler()
	{
		assert(_deadlineTasks != nullptr);
		delete _deadlineTasks;
	}

	//! \brief Get a ready task for execution
	//!
	//! \param[in] computePlace The hardware place asking for scheduling orders
	//! incompatible with the computePlace asking
	//!
	//! \returns A ready task or nullptr
	Task *getReadyTask(ComputePlace *computePlace);
};

#endif // HOST_UNSYNC_SCHEDULER_HPP
