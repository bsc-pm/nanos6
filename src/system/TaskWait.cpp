/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include <nanos6.h>

#include "DataAccessRegistration.hpp"
#include "TaskBlocking.hpp"
#include "TaskWait.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/device/directory/Directory.hpp"
#include "hardware/HardwareInfo.hpp"
#include "system/TrackingPoints.hpp"
#include "tasks/StreamManager.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"

#include <InstrumentTaskStatus.hpp>


void nanos6_taskwait(char const *invocationSource)
{
	TaskWait::taskWait(invocationSource, true);
}

void TaskWait::taskWait(char const *invocationSource, bool fromUserCode)
{
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);

	Task *currentTask = currentThread->getTask();
	assert(currentTask != nullptr);
	assert(currentTask->getThread() == currentThread);

	// Runtime Tracking Point - Entering a taskwait, the task will be blocked
	TrackingPoints::enterTaskWait(currentTask, invocationSource, fromUserCode);

	// Fast check
	// Note that this must only be used with
	if (currentTask->doesNotNeedToBlockForChildren() && !Directory::isEnabled()) {
		// This in combination with a release from the children makes their changes visible to this thread
		std::atomic_thread_fence(std::memory_order_acquire);

		// Runtime Tracking Point - Exiting a taskwait, the task will be resumed
		TrackingPoints::exitTaskWait(currentTask, fromUserCode);
		return;
	}

	ComputePlace *cpu = currentThread->getComputePlace();
	assert(cpu != nullptr);

	DataAccessRegistration::handleEnterTaskwait(currentTask, cpu, cpu->getDependencyData());
	bool done = currentTask->markAsBlocked();

	// done == true:
	//   1. The condition of the taskwait has been fulfilled
	//   2. The task will not be queued at all
	//   3. The execution must continue (without blocking)
	// done == false:
	//   1. The task has been marked as blocked
	//   2. At any time the condition of the taskwait can become true
	//   3. The thread responsible for that change will queue the task
	//   4. Any thread can dequeue it and attempt to resume the thread
	//   5. This can trigger a migration, and will make the call to
	//     ThreadManager::switchThreads (that is inside TaskBlocking::taskBlocks)
	//     to resume immediately (and to wake the replacement thread, if any,
	//     on the "old" CPU)
	Instrument::task_id_t taskId = currentTask->getInstrumentationTaskId();
	if (!done) {
		Instrument::taskIsBlocked(taskId, Instrument::in_taskwait_blocking_reason);

		TaskBlocking::taskBlocks(currentThread, currentTask);

		// Update the CPU since the thread may have migrated
		cpu = currentThread->getComputePlace();
		assert(cpu != nullptr);

		Instrument::ThreadInstrumentationContext::updateComputePlace(cpu->getInstrumentationId());
	}

	// This in combination with a release from the children makes their changes visible to this thread
	std::atomic_thread_fence(std::memory_order_acquire);

	// Now that everything is finished, we actually need to perform any needed directory flushes
	// There are two possible cases: in the top-level main task, we will flush everything
	// Otherwise, we will only flush the task's dependencies

	bool copiesFinished = false;
	if (currentTask->getParent() == nullptr) {
		copiesFinished = Directory::flush(currentTask);
	} else {
		copiesFinished = Directory::flushTaskDependencies(currentTask);
	}

	// Block again until all directory copies are done
	if (!copiesFinished) {
		Instrument::taskIsBlocked(taskId, Instrument::in_taskwait_blocking_reason);

		TaskBlocking::taskBlocks(currentThread, currentTask);

		// Update the CPU since the thread may have migrated
		cpu = currentThread->getComputePlace();
		assert(cpu != nullptr);

		Instrument::ThreadInstrumentationContext::updateComputePlace(cpu->getInstrumentationId());

		std::atomic_thread_fence(std::memory_order_acquire);
	}

	assert(currentTask->canBeWokenUp());
	currentTask->markAsUnblocked();

	DataAccessRegistration::handleExitTaskwait(currentTask, cpu, cpu->getDependencyData());

	if (!done) {
		// The instrumentation was notified that the task had been blocked
		Instrument::taskIsExecuting(taskId, true);
	}

	// Runtime Tracking Point - Exiting a taskwait, the task will be resumed
	TrackingPoints::exitTaskWait(currentTask, fromUserCode);
}

void nanos6_stream_synchronize(size_t stream_id)
{
	StreamManager::synchronizeStream(stream_id);
}

void nanos6_stream_synchronize_all(void)
{
	StreamManager::synchronizeAllStreams();
}

