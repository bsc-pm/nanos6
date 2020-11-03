/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef IF0_TASK_HPP
#define IF0_TASK_HPP

#include <cassert>

#include "TrackingPoints.hpp"
#include "executors/threads/CPU.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/Task.hpp"

#include <InstrumentThreadManagement.hpp>


class ComputePlace;


namespace If0Task {

	//! \brief Waits for the child if(0) task to finish.
	//!
	//! This function will lock the task by replacing it in the current thread, but as Task::markAsBlocked is
	//! not called, the only way to unlock this task is to put it directly in the scheduler. The if(0) task
	//! is in charge of doing this after execution.
	inline void waitForIf0Task(WorkerThread *currentThread, Task *currentTask, Task *if0Task, ComputePlace *computePlace)
	{
		assert(currentThread != nullptr);
		assert(currentTask != nullptr);
		assert(if0Task != nullptr);
		assert(computePlace != nullptr);

		CPU *cpu = static_cast<CPU *>(computePlace);
		assert(cpu != nullptr);

		// Runtime Tracking Point - Entering a taskwait through If0, the task will be blocked
		TrackingPoints::enterWaitForIf0Task(currentTask, if0Task, currentThread, cpu);

		WorkerThread *replacementThread = ThreadManager::getIdleThread(cpu);
		currentThread->switchTo(replacementThread);

		// Update the CPU since the thread may have migrated
		cpu = currentThread->getComputePlace();
		assert(cpu != nullptr);

		Instrument::ThreadInstrumentationContext::updateComputePlace(cpu->getInstrumentationId());

		// Runtime Tracking Point - Exiting a taskwait through If0, the task will resume
		TrackingPoints::exitWaitForIf0Task(currentTask);
	}


	inline void executeInline(
		WorkerThread *currentThread, Task *currentTask, Task *if0Task,
		ComputePlace *computePlace
	) {
		assert(currentThread != nullptr);
		assert(currentTask != nullptr);
		assert(if0Task != nullptr);
		assert(if0Task->getParent() == currentTask);
		assert(computePlace != nullptr);

		// Runtime Tracking Point - Entering a taskwait through If0, the task will be blocked
		TrackingPoints::enterExecuteInline(currentTask, if0Task);

		currentThread->handleTask((CPU *) computePlace, if0Task);

		// Runtime Tracking Point - Exiting a taskwait (from If0), the task will be resumed
		TrackingPoints::exitExecuteInline(currentTask, if0Task);
	}


	inline void executeNonInline(
		WorkerThread *currentThread, Task *if0Task,
		ComputePlace *computePlace
	) {
		assert(currentThread != nullptr);
		assert(computePlace != nullptr);
		assert(if0Task != nullptr);
		assert(if0Task->isIf0());

		Task *parent = if0Task->getParent();
		assert(parent != nullptr);

		currentThread->handleTask((CPU *) computePlace, if0Task);

		// The thread can migrate during the execution of the task
		computePlace = currentThread->getComputePlace();

		Scheduler::addReadyTask(parent, computePlace, UNBLOCKED_TASK_HINT);
	}

}


#endif // IF0_TASK_HPP
