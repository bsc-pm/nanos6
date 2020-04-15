/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include <nanos6/blocking.h>

#include "DataAccessRegistration.hpp"
#include "executors/threads/CPUManager.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware-counters/HardwareCounters.hpp"
#include "ompss/TaskBlocking.hpp"
#include "scheduling/Scheduler.hpp"

#include <InstrumentBlocking.hpp>
#include <Monitoring.hpp>


extern "C" void *nanos6_get_current_blocking_context(void)
{
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);

	Task *currentTask = currentThread->getTask();
	assert(currentTask != nullptr);

	return currentTask;
}


extern "C" void nanos6_block_current_task(__attribute__((unused)) void *blocking_context)
{
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);

	Task *currentTask = currentThread->getTask();
	assert(currentTask != nullptr);

	assert(blocking_context == currentTask);

	Monitoring::taskChangedStatus(currentTask, blocked_status);
	HardwareCounters::taskStopped(currentTask);

	Instrument::taskIsBlocked(currentTask->getInstrumentationTaskId(), Instrument::user_requested_blocking_reason);
	Instrument::enterBlocking(currentTask->getInstrumentationTaskId());

	TaskBlocking::taskBlocks(currentThread, currentTask);

	ComputePlace *computePlace = currentThread->getComputePlace();
	assert(computePlace != nullptr);
	Instrument::ThreadInstrumentationContext::updateComputePlace(computePlace->getInstrumentationId());

	Instrument::exitBlocking(currentTask->getInstrumentationTaskId());
	Instrument::taskIsExecuting(currentTask->getInstrumentationTaskId());

	HardwareCounters::taskStarted(currentTask);
	Monitoring::taskChangedStatus(currentTask, executing_status);
}


extern "C" void nanos6_unblock_task(void *blocking_context)
{
	Task *task = static_cast<Task *>(blocking_context);

	Instrument::unblockTask(task->getInstrumentationTaskId());
	Scheduler::addReadyTask(task, nullptr, UNBLOCKED_TASK_HINT);

	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	ComputePlace *computePlace = nullptr;
	if (currentThread != nullptr) {
		computePlace = currentThread->getComputePlace();
	}

	// After adding a task, the CPUManager may want to unidle CPUs
	CPUManager::executeCPUManagerPolicy(computePlace, ADDED_TASKS, 1);
}

