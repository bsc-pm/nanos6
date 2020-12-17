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

class ReadyQueueDeque;
class ReadyQueueMap;

class UnsyncScheduler {
protected:
	typedef Container::vector<Task *> immediate_successor_tasks_t;

	immediate_successor_tasks_t _immediateSuccessorTasks;
	immediate_successor_tasks_t _immediateSuccessorTaskfors;

	ReadyQueue **_queues;
	size_t _numQueues;
	DeadlineQueue *_deadlineTasks;
	// When tasks does not have a NUMA hint, we assign it in a round robin basis.
	uint64_t _roundRobinQueues;

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
						regularAddReadyTask(currentIS, hint == UNBLOCKED_TASK_HINT);
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
						regularAddReadyTask(currentIS1, hint == UNBLOCKED_TASK_HINT);
						_immediateSuccessorTaskfors[immediateSuccessorId] = task;
					}
				}
				return;
			}
		}

		regularAddReadyTask(task, hint);
	}

	//! \brief Get a ready task for execution
	//!
	//! \param[in] computePlace the hardware place asking for scheduling orders
	//!
	//! \returns a ready task or nullptr
	virtual Task *getReadyTask(ComputePlace *computePlace) = 0;
protected:
	void regularAddReadyTask(Task *task, bool unblocked);

	Task *regularGetReadyTask(ComputePlace *computePlace);

	virtual bool enableNUMA()
	{
		return true;
	}
};


#endif // UNSYNC_SCHEDULER_HPP
