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
#include "hardware/HardwareInfo.hpp"
#include "hardware-counters/HardwareCounters.hpp"
#include "monitoring/Monitoring.hpp"
#include "tasks/StreamManager.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"

#include <InstrumentTaskStatus.hpp>
#include <InstrumentTaskWait.hpp>


void nanos6_taskwait(char const *invocationSource)
{
	TaskWait::taskWait(invocationSource, true);
}

void TaskWait::taskWait(char const *invocationSource, bool fromUserCode)
{
	Task *currentTask = nullptr;
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();

	assert(currentThread != nullptr);

	currentTask = currentThread->getTask();
	assert(currentTask != nullptr);
	assert(currentTask->getThread() == currentThread);

	if (fromUserCode) {
		HardwareCounters::updateTaskCounters(currentTask);
		Monitoring::taskChangedStatus(currentTask, paused_status);
	}
	Instrument::task_id_t taskId = currentTask->getInstrumentationTaskId();
	Instrument::enterTaskWait(taskId, invocationSource, Instrument::task_id_t(), fromUserCode);

	// Fast check
	if (currentTask->doesNotNeedToBlockForChildren()) {
		// This in combination with a release from the children makes their changes visible to this thread
		std::atomic_thread_fence(std::memory_order_acquire);

		if (fromUserCode) {
			HardwareCounters::updateRuntimeCounters();
			Monitoring::taskChangedStatus(currentTask, executing_status);
		}
		Instrument::exitTaskWait(taskId, fromUserCode);

		return;
	}

	ComputePlace *cpu = currentThread->getComputePlace();
	assert(cpu != nullptr);
	DataAccessRegistration::handleEnterTaskwait(currentTask, cpu, cpu->getDependencyData());
	bool done = currentTask->markAsBlocked();

	// done == true:
	// 	1. The condition of the taskwait has been fulfilled
	// 	2. The task will not be queued at all
	// 	3. The execution must continue (without blocking)
	// done == false:
	// 	1. The task has been marked as blocked
	// 	2. At any time the condition of the taskwait can become true
	// 	3. The thread responsible for that change will queue the task
	// 	4. Any thread can dequeue it and attempt to resume the thread
	// 	5. This can trigger a migration, and will make the call to
	// 		ThreadManager::switchThreads (that is inside TaskBlocking::taskBlocks)
	// 		to resume immediately (and to wake the replacement thread, if any,
	// 		on the "old" CPU)

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

	assert(currentTask->canBeWokenUp());
	currentTask->markAsUnblocked();

	DataAccessRegistration::handleExitTaskwait(currentTask, cpu, cpu->getDependencyData());

	if (!done) {
		// The instrumentation was notified that the task had been blocked
		Instrument::taskIsExecuting(taskId, true);
	}

	if (fromUserCode) {
		HardwareCounters::updateRuntimeCounters();
		Instrument::exitTaskWait(taskId, fromUserCode);
		Monitoring::taskChangedStatus(currentTask, executing_status);
	} else {
		Instrument::exitTaskWait(taskId, fromUserCode);
	}
}

void nanos6_stream_synchronize(size_t stream_id)
{
	StreamManager::synchronizeStream(stream_id);
}

void nanos6_stream_synchronize_all(void)
{
	StreamManager::synchronizeAllStreams();
}

