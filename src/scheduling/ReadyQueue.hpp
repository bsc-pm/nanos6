/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef READY_QUEUE_HPP
#define READY_QUEUE_HPP

class ComputePlace;
class Task;

enum SchedulingPolicy {
	LIFO_POLICY,
	FIFO_POLICY
};

enum ReadyTaskHint {
	NO_HINT,
	CHILD_TASK_HINT,
	SIBLING_TASK_HINT,
	BUSY_COMPUTE_PLACE_TASK_HINT,
	UNBLOCKED_TASK_HINT
};

//! \brief Interface that ready queues must implement
class ReadyQueue {
protected:
	SchedulingPolicy _policy;

public:
	ReadyQueue(SchedulingPolicy policy)
		: _policy(policy)
	{}

	virtual ~ReadyQueue()
	{}

	//! \brief Add a (ready) task that has been created or freed
	//!
	//! \param[in] task the task to be added
	//! \param[in] unblocked whether it is an unblocked task or not
	virtual void addReadyTask(Task *task, bool unblocked) = 0;

	//! \brief Get a ready task for execution
	//!
	//! \returns a ready task or nullptr
	virtual Task *getReadyTask(ComputePlace *computePlace) = 0;

	virtual size_t getNumReadyTasks() const = 0;
};


#endif // READY_QUEUE_HPP
