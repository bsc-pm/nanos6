/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <cstdint>

#include <nanos6/blocking.h>

#include "DataAccessRegistration.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware-counters/HardwareCounters.hpp"
#include "ompss/TaskBlocking.hpp"
#include "scheduling/Scheduler.hpp"
#include "support/chronometers/std/Chrono.hpp"

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

	HardwareCounters::readTaskCounters(currentTask);
	Monitoring::taskChangedStatus(currentTask, blocked_status);
	Instrument::taskIsBlocked(currentTask->getInstrumentationTaskId(), Instrument::user_requested_blocking_reason);
	Instrument::enterBlocking(currentTask->getInstrumentationTaskId());

	TaskBlocking::taskBlocks(currentThread, currentTask);

	ComputePlace *computePlace = currentThread->getComputePlace();
	assert(computePlace != nullptr);
	Instrument::ThreadInstrumentationContext::updateComputePlace(computePlace->getInstrumentationId());

	HardwareCounters::readCPUCounters();
	Instrument::exitBlocking(currentTask->getInstrumentationTaskId());
	Instrument::taskIsExecuting(currentTask->getInstrumentationTaskId());
	Monitoring::taskChangedStatus(currentTask, executing_status);
}


extern "C" void nanos6_unblock_task(void *blocking_context)
{
	Task *task = static_cast<Task *>(blocking_context);

	Instrument::unblockTask(task->getInstrumentationTaskId());

	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	ComputePlace *computePlace = nullptr;
	if (currentThread != nullptr) {
		computePlace = currentThread->getComputePlace();
	}

	Scheduler::addReadyTask(task, computePlace, UNBLOCKED_TASK_HINT);
}


extern "C" uint64_t nanos6_wait_for(uint64_t time_us)
{
	if (time_us == 0)
		return 0;

	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);

	CPU *cpu = currentThread->getComputePlace();
	assert(cpu != nullptr);

	Task *currentTask = currentThread->getTask();
	assert(currentTask != nullptr);

	Task::deadline_t timeout = (Task::deadline_t) time_us;

	// Substract a fixed delta to the timeout
	const Task::deadline_t minimumCost = 30;
	if (timeout > minimumCost) {
		timeout -= minimumCost;
	} else {
		timeout = 0;
	}

	// Update the task deadline
	Task::deadline_t start = Chrono::now<Task::deadline_t>();
	currentTask->setDeadline(start + timeout);

	HardwareCounters::taskStopped(currentTask);

	// Re-add the current task to the scheduler with a deadline
	Scheduler::addReadyTask(currentTask, cpu, DEADLINE_TASK_HINT);

	TaskBlocking::taskBlocks(currentThread, currentTask);

	HardwareCounters::taskStarted(currentTask);
	Monitoring::taskChangedStatus(currentTask, executing_status);

	// Update the CPU since the thread may have migrated
	cpu = currentThread->getComputePlace();
	assert(cpu != nullptr);
	Instrument::ThreadInstrumentationContext::updateComputePlace(cpu->getInstrumentationId());

	return (uint64_t) (Chrono::now<Task::deadline_t>() - start);
}
