/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef SCHEDULER_HPP
#define SCHEDULER_HPP

#include "SchedulerInterface.hpp"
#include "system/TrackingPoints.hpp"
#include "tasks/Task.hpp"

#include <InstrumentScheduler.hpp>


class Scheduler {
	static SchedulerInterface *_instance;

public:
	static void initialize();
	static void shutdown();

	//! \brief Add multiple ready tasks to the scheduler
	//!
	//! \param taskType the task type of the tasks
	//! \param tasks the array of the tasks to be added
	//! \param numTasks the number of tasks to add
	//! \param computePlace the compute place from where the tasks are added
	//! \param hint the scheduling hint of the tasks
	static inline void addReadyTasks(
		nanos6_device_t taskType, Task *tasks[],
		const size_t numTasks, ComputePlace *computePlace,
		ReadyTaskHint hint
	) {
		assert(computePlace == nullptr || computePlace->getType() == nanos6_host_device);

		// Runtime Tracking Point - Tasks will be added to the scheduler and will be ready
		TrackingPoints::enterAddReadyTasks(tasks, numTasks);

		_instance->addReadyTasks(taskType, tasks, numTasks, computePlace, hint);

		// Runtime Tracking Point - Exiting the addReadyTasks function
		TrackingPoints::exitAddReadyTasks();
	}

	//! \brief Add a ready task to the scheduler
	//!
	//! \param task the ready task to add
	//! \param computePlace the compute place from where the task is added
	//! \param hint the scheduling hint of the task
	static inline void addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint = NO_HINT)
	{
		assert(computePlace == nullptr || computePlace->getType() == nanos6_host_device);

		// Runtime Tracking Point - A task will be added to the scheduler and will be readys
		TrackingPoints::enterAddReadyTask(task);

		_instance->addReadyTask(task, computePlace, hint);

		// Runtime Tracking Point - Exiting the addReadyTasks function
		TrackingPoints::exitAddReadyTask();
	}

	//! \brief Get a ready task from the scheduler
	//!
	//! This function tries to get a ready task from the scheduler. Tasks may have
	//! onready actions to execute, so this is the point where we execute them. Thus,
	//! a thread getting a ready task may need to execute a few onready actions before
	//! actually getting any task. In case a thread should not execute several onready
	//! actions (e.g., it is busy doing other work), it can pass true to the parameter
	//! fromBusyThread
	//!
	//! \param computePlace the target compute place that wants to execute a task
	//! \param currentThread the current running thread
	//! \param fromBusyThread whether the thread should not run multiple onready actions
	//!
	//! \returns the ready task or nullptr
	static inline Task *getReadyTask(ComputePlace *computePlace, WorkerThread *currentThread, bool fromBusyThread = false)
	{
		assert(computePlace != nullptr);
		assert(currentThread != nullptr);

		Task *task = nullptr;

		bool retry;
		do {
			Instrument::enterGetReadyTask();
			task = _instance->getReadyTask(computePlace);
			Instrument::exitGetReadyTask();

			retry = false;
			if (task != nullptr && !task->handleOnready(currentThread)) {
				retry = !fromBusyThread;
				task = nullptr;
			}
		} while (retry);

		return task;
	}

	//! \brief Check whether a compute place is serving tasks
	//!
	//! This function is called to check whether there is any compute place serving
	//! tasks. Notice that we require to have a compute place serving tasks except
	//! when there is work for all compute places. This information is considered
	//! when a compute place is about to be marked as idle. It should abort the idle
	//! process when detecting that there are no compute places serving tasks
	static inline bool isServingTasks()
	{
		return _instance->isServingTasks();
	}

	//! \brief Check whether task priority is considered
	static inline bool isPriorityEnabled()
	{
		return SchedulerInterface::isPriorityEnabled();
	}

	static inline float getImmediateSuccessorAlpha()
	{
		return SchedulerInterface::getImmediateSuccessorAlpha();
	}
};

#endif // SCHEDULER_HPP
