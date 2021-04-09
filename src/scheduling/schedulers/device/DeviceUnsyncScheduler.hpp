/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef DEVICE_UNSYNC_SCHEDULER_HPP
#define DEVICE_UNSYNC_SCHEDULER_HPP

#include "scheduling/ready-queues/ReadyQueueDeque.hpp"
#include "scheduling/ready-queues/ReadyQueueMap.hpp"
#include "scheduling/schedulers/UnsyncScheduler.hpp"

class DeviceUnsyncScheduler : public UnsyncScheduler {
public:
	DeviceUnsyncScheduler(SchedulingPolicy policy, bool enablePriority)
		: UnsyncScheduler(policy, enablePriority)
	{
		_numQueues = 1;
		_queues = (ReadyQueue **) MemoryAllocator::alloc(sizeof(ReadyQueue *));

		// Create a single queue at the first position
		if (enablePriority) {
			_queues[0] = new ReadyQueueMap(policy);
		} else {
			_queues[0] = new ReadyQueueDeque(policy);
		}
	}

	virtual ~DeviceUnsyncScheduler()
	{
	}

	//! \brief Get a ready task for execution
	//!
	//! \param[in] computePlace the hardware place asking for scheduling orders
	//! \param[out] hasIncompatibleWork Whether the scheduler has work available but
	//! incompatible with the computePlace asking
	//!
	//! \returns a ready task or nullptr
	Task *getReadyTask(ComputePlace *computePlace, bool &hasIncompatibleWork);
};


#endif // DEVICE_UNSYNC_SCHEDULER_HPP
