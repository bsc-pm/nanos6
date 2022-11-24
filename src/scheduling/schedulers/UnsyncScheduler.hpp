/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef UNSYNC_SCHEDULER_HPP
#define UNSYNC_SCHEDULER_HPP

#include <cassert>

#include "hardware/places/ComputePlace.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "scheduling/ReadyQueue.hpp"
#include "scheduling/ready-queues/DeadlineQueue.hpp"
#include "support/Containers.hpp"
#include "tasks/Task.hpp"


class UnsyncScheduler {
protected:
	ReadyQueue **_queues;
	size_t _numQueues;

	// When tasks do not have a NUMA hints we assign them in a round robin basis
	uint64_t _roundRobinQueues;

	DeadlineQueue *_deadlineTasks;

	bool _enablePriority;

public:
	UnsyncScheduler(SchedulingPolicy policy, bool enablePriority);

	virtual ~UnsyncScheduler();

	//! \brief Add a (ready) task that has been created or freed
	//!
	//! \param[in] task the task to be added
	//! \param[in] computePlace the hardware place of the creator or the liberator
	//! \param[in] hint a hint about the relation of the task to the current task
	virtual inline void addReadyTask(Task *task, ComputePlace *, ReadyTaskHint hint = NO_HINT)
	{
		assert(task != nullptr);

		if (hint == DEADLINE_TASK_HINT) {
			assert(task->hasDeadline());
			assert(_deadlineTasks != nullptr);

			_deadlineTasks->addReadyTask(task, true);
			return;
		}

		regularAddReadyTask(task, hint == UNBLOCKED_TASK_HINT);
	}

	//! \brief Get a ready task for execution
	//!
	//! \param[in] computePlace the hardware place asking for scheduling orders
	//! incompatible with the computePlace asking
	//!
	//! \returns a ready task or nullptr
	virtual Task *getReadyTask(ComputePlace *computePlace) = 0;

protected:
	//! \brief Add ready task considering NUMA queues
	//!
	//! \param[in] task the ready task to add
	//! \param[in] unblocked whether it is an unblocked task or not
	void regularAddReadyTask(Task *task, bool unblocked);

	//! \brief Get a ready task considering NUMA queues
	//!
	//! \param[in] computePlace the hardware place asking for scheduling orders
	//!
	//! \returns a ready task or nullptr
	Task *regularGetReadyTask(ComputePlace *computePlace);
};


#endif // UNSYNC_SCHEDULER_HPP
