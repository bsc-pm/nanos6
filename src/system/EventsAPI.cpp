/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2023 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include "CPUDependencyData.hpp"
#include "DataAccessRegistration.hpp"
#include "EventsAPI.hpp"
#include "executors/threads/TaskFinalization.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "executors/threads/WorkerThreadImplementation.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"


void EventsAPI::increaseCurrentTaskEvents(unsigned int increment)
{
	if (increment == 0)
		return;

	Task *task = WorkerThread::getCurrentTask();
	assert(task != nullptr);

	task->increaseReleaseCount(increment);
}

void EventsAPI::decreaseTaskEvents(Task *task, unsigned int decrement)
{
	if (decrement == 0)
		return;

	// Decrease the release count
	if (!task->decreaseReleaseCount(decrement))
		return;

	CPU *cpu = nullptr;
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	if (currentThread != nullptr) {
		cpu = currentThread->getComputePlace();
		assert(cpu != nullptr);
	}

	if (!task->isOnreadyCompleted()) {
		// All onready events completed and the task is ready to execute
		task->completeOnready();

		Scheduler::addReadyTask(task, cpu, UNBLOCKED_TASK_HINT);
	} else {
		// Release the data accesses of the task. Do not merge these
		// two conditions; the creation of a local CPU dependency data
		// structure may introduce unnecessary overhead
		if (cpu != nullptr) {
			DataAccessRegistration::unregisterTaskDataAccesses(
				task, cpu, cpu->getDependencyData(),
				/* memory place */ nullptr,
				/* from a busy thread */ true
			);
		} else {
			CPUDependencyData localDependencyData;
			DataAccessRegistration::unregisterTaskDataAccesses(
				task, nullptr, localDependencyData,
				/* memory place */ nullptr,
				/* from a busy thread */ true
			);
		}

		TaskFinalization::taskFinished(task, cpu, true);

		// Try to dispose the task
		if (task->markAsReleased())
			TaskFinalization::disposeTask(task);
	}
}

extern "C" void *nanos6_get_current_event_counter(void)
{
	return WorkerThread::getCurrentTask();
}

extern "C" void nanos6_increase_current_task_event_counter(void *, unsigned int increment)
{
	EventsAPI::increaseCurrentTaskEvents(increment);
}

extern "C" void nanos6_decrease_task_event_counter(void *event_counter, unsigned int decrement)
{
	Task *task = static_cast<Task *>(event_counter);
	assert(task != nullptr);

	EventsAPI::decreaseTaskEvents(task, decrement);
}
