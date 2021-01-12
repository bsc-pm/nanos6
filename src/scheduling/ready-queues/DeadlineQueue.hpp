/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef DEADLINE_QUEUE_HPP
#define DEADLINE_QUEUE_HPP

#include "scheduling/ReadyQueue.hpp"
#include "support/Chrono.hpp"
#include "support/Containers.hpp"
#include "tasks/Task.hpp"

//! This kind of ready queue supports deadlines
class DeadlineQueue : public ReadyQueue {
	//! Deadline compare function
	struct DeadlineCompare {
		bool operator() (const Task *left, const Task *right) const
		{
			if (left->getDeadline() != right->getDeadline()) {
				return (left->getDeadline() < right->getDeadline());
			}
			return (left < right);
		}
	};

	//! Set of deadline tasks ordered by their task deadline
	//! in ascending order. Tasks with the lowest deadline
	//! are stored at the first positions. It also allows
	//! different tasks to share the same deadline
	typedef Container::set<Task *, DeadlineCompare> deadline_set_t;

	//! The ordered set of deadline tasks
	deadline_set_t _deadlineTasks;

	//! The cached current time point (may be stale)
	mutable Task::deadline_t _now;

public:
	inline DeadlineQueue(SchedulingPolicy policy) :
		ReadyQueue(policy),
		_deadlineTasks(),
		_now(Chrono::now<Task::deadline_t>())
	{
	}

	inline ~DeadlineQueue()
	{
		assert(_deadlineTasks.empty());
	}

	//! \brief Add ready task with deadline
	//!
	//! The task is ready but it has been paused by the user
	//! with a timeout. Thus, we should not resume the task
	//! until the deadline is approaching or surpassed
	//!
	//! \param task The deadline task
	//! \param unblocked Whether the task is unblocked
	inline void addReadyTask(Task *task, bool)
	{
		assert(task->hasDeadline());

#ifndef NDEBUG
		auto res = _deadlineTasks.insert(task);
		assert(res.second);
#else
		_deadlineTasks.insert(task);
#endif
	}

	//! \brief Get a ready task with the deadline satisfied
	//!
	//! \param computePlace The current compute place
	inline Task *getReadyTask(ComputePlace *)
	{
		if (_deadlineTasks.empty())
			return nullptr;

		auto it = _deadlineTasks.begin();
		assert(it != _deadlineTasks.end());

		Task *task = *it;
		assert(task != nullptr);

		// First check using the cached current time
		// and then using the updated current time
		if (task->getDeadline() <= _now) {
			_deadlineTasks.erase(it);
			return task;
		} else {
			_now = Chrono::now<Task::deadline_t>();
			if (task->getDeadline() < _now) {
				_deadlineTasks.erase(it);
				return task;
			}
		}
		return nullptr;
	}

	//! \brief Get the number of available deadline tasks
	inline size_t getNumReadyTasks() const
	{
		if (_deadlineTasks.empty())
			return 0;

		_now = Chrono::now<Task::deadline_t>();

		size_t numReady = 0;
		for (Task *task : _deadlineTasks) {
			if (task->getDeadline() > _now)
				break;

			++numReady;
		}
		return numReady;
	}
};


#endif // DEADLINE_QUEUE_HPP
