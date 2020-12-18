/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef UNSYNC_SCHEDULER_HPP
#define UNSYNC_SCHEDULER_HPP

#include <cassert>

#include "hardware/places/ComputePlace.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "scheduling/ready-queues/DeadlineQueue.hpp"
#include "support/Containers.hpp"
#include "tasks/Task.hpp"


class UnsyncScheduler {
protected:
	typedef Container::vector<Task *> immediate_successor_tasks_t;

	immediate_successor_tasks_t _immediateSuccessorTasks;
	immediate_successor_tasks_t _immediateSuccessorTaskfors;

	ReadyQueue **_queues;
	size_t _numQueues;

	// When tasks do not have a NUMA hints we assign them in a round robin basis
	uint64_t _roundRobinQueues;

	DeadlineQueue *_deadlineTasks;

	bool _enableImmediateSuccessor;
	bool _enablePriority;

public:
	UnsyncScheduler(SchedulingPolicy policy, bool enablePriority, bool enableImmediateSuccessor);

	virtual ~UnsyncScheduler();

	//! \brief Add a (ready) task that has been created or freed
	//!
	//! \param[in] task the task to be added
	//! \param[in] computePlace the hardware place of the creator or the liberator
	//! \param[in] hint a hint about the relation of the task to the current task
	virtual inline void addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint = NO_HINT)
	{
		assert(task != nullptr);

		if (hint == DEADLINE_TASK_HINT) {
			assert(task->hasDeadline());
			assert(_deadlineTasks != nullptr);

			_deadlineTasks->addReadyTask(task, true);
			return;
		}

		if (_enableImmediateSuccessor) {
			if (computePlace != nullptr && hint == SIBLING_TASK_HINT) {
				size_t immediateSuccessorId = computePlace->getIndex();
				if (!task->isTaskfor()) {
					Task *currentIS = _immediateSuccessorTasks[immediateSuccessorId];
					if (currentIS != nullptr) {
						assert(!currentIS->isTaskfor());
						regularAddReadyTask(currentIS, false);
					}
					_immediateSuccessorTasks[immediateSuccessorId] = task;
				} else {
					// Multiply by 2 because there are 2 slots per group
					immediateSuccessorId = ((CPU *)computePlace)->getGroupId()*2;
					Task *currentIS1 = _immediateSuccessorTaskfors[immediateSuccessorId];
					Task *currentIS2 = _immediateSuccessorTaskfors[immediateSuccessorId+1];
					if (currentIS1 == nullptr) {
						_immediateSuccessorTaskfors[immediateSuccessorId] = task;
					} else if (currentIS2 == nullptr) {
						_immediateSuccessorTaskfors[immediateSuccessorId+1] = task;
					} else {
						regularAddReadyTask(currentIS1, false);
						_immediateSuccessorTaskfors[immediateSuccessorId] = task;
					}
				}
				return;
			}
		}

		regularAddReadyTask(task, hint == UNBLOCKED_TASK_HINT);
	}

	//! \brief Get a ready task for execution
	//!
	//! \param[in] computePlace the hardware place asking for scheduling orders
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

	virtual bool enableNUMA()
	{
		return true;
	}
};


#endif // UNSYNC_SCHEDULER_HPP
