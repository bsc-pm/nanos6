/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef UNSYNC_SCHEDULER_HPP
#define UNSYNC_SCHEDULER_HPP

#include <cassert>
#include <vector>

#include "hardware/places/ComputePlace.hpp"
#include "scheduling/ReadyQueue.hpp"
#include "tasks/Task.hpp"


class UnsyncScheduler {
protected:
	std::vector<Task *> _immediateSuccessorTasks;
	std::vector<Task *> _immediateSuccessorTaskfors;
	ReadyQueue *_readyTasks;
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

		bool unblocked = (hint == UNBLOCKED_TASK_HINT);
		if (_enableImmediateSuccessor) {
			if (computePlace != nullptr && hint == SIBLING_TASK_HINT) {
				size_t immediateSuccessorId = computePlace->getIndex();
				if (!task->isTaskfor()) {
					Task *currentIS = _immediateSuccessorTasks[immediateSuccessorId];
					if (currentIS != nullptr) {
						assert(!currentIS->isTaskfor());
						_readyTasks->addReadyTask(currentIS, false);
					}
					_immediateSuccessorTasks[immediateSuccessorId] = task;
				} else {
					// Multiply by 2 because there are 2 slots per group
					immediateSuccessorId = ((CPU *)computePlace)->getGroupId()*2;
					Task *currentIS1 = _immediateSuccessorTaskfors[immediateSuccessorId];
					Task *currentIS2 = _immediateSuccessorTaskfors[immediateSuccessorId+1];
					if (currentIS1 == nullptr) {
						_immediateSuccessorTaskfors[immediateSuccessorId] = task;
					}
					else if (currentIS2 == nullptr) {
						_immediateSuccessorTaskfors[immediateSuccessorId+1] = task;
					}
					else {
						_readyTasks->addReadyTask(currentIS1, false);
						_immediateSuccessorTaskfors[immediateSuccessorId] = task;
					}
				}
				return;
			}
		}

		_readyTasks->addReadyTask(task, unblocked);
	}

	//! \brief Get a ready task for execution
	//!
	//! \param[in] computePlace the hardware place asking for scheduling orders
	//!
	//! \returns a ready task or nullptr
	virtual Task *getReadyTask(ComputePlace *computePlace) = 0;

	//! \brief Check if the scheduler has available work for the current CPU
	//!
	//! \param[in] computePlace The host compute place
	virtual bool hasAvailableWork(ComputePlace *computePlace) = 0;

	//! \brief Notify the scheduler that a CPU is about to be disabled
	//! in case any tasks must be unassigned
	//!
	//! \param[in] cpuId The id of the cpu that will be disabled
	//! \param[in] task A task assigned to the current thread or nullptr
	//!
	//! \return Whether work was reassigned upon disabling the CPU
	inline bool disablingCPU(size_t cpuId, Task *task)
	{
		// If the current thread had a task assigned, readd it to the scheduler
		if (task != nullptr) {
			_readyTasks->addReadyTask(task, false);
		}

		if (_enableImmediateSuccessor) {
			// Upon disabling a CPU, if its immediate successor slot was full
			// place the task in the ready queue
			Task *currentIS = _immediateSuccessorTasks[cpuId];
			if (currentIS != nullptr) {
				_immediateSuccessorTasks[cpuId] = nullptr;
				_readyTasks->addReadyTask(currentIS, false);
				return true;
			}
		}

		return false;
	}
};


#endif // UNSYNC_SCHEDULER_HPP
