/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <cstdint>

#include <nanos6/blocking.h>

#include "DataAccessRegistration.hpp"
#include "TrackingPoints.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "ompss/TaskBlocking.hpp"
#include "scheduling/Scheduler.hpp"
#include "support/chronometers/std/Chrono.hpp"


void BlockingAPI::blockCurrentTask(bool fromUserCode)
{
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);

	Task *currentTask = currentThread->getTask();
	assert(currentTask != nullptr);

	// Runtime Tracking Point - The current task is gonna be blocked
	TrackingPoints::enterBlockCurrentTask(currentTask, fromUserCode);

	TaskBlocking::taskBlocks(currentThread, currentTask);

	ComputePlace *computePlace = currentThread->getComputePlace();
	assert(computePlace != nullptr);

	// Update the CPU as this thread may have migrated
	Instrument::ThreadInstrumentationContext::updateComputePlace(computePlace->getInstrumentationId());

	// Runtime Tracking Point - The current task resumes its execution
	TrackingPoints::exitBlockCurrentTask(currentTask, fromUserCode);
}

void BlockingAPI::unblockTask(Task *task, bool fromUserCode)
{
	assert(task != nullptr);

	Task *currentTask = nullptr;
	ComputePlace *computePlace = nullptr;
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	if (currentThread != nullptr) {
		currentTask = currentThread->getTask();
		computePlace = currentThread->getComputePlace();
	}

	// Runtime Tracking Point - The current task is gonna execute runtime code
	TrackingPoints::enterUnblockTask(task, currentTask, fromUserCode);

	Scheduler::addReadyTask(task, computePlace, UNBLOCKED_TASK_HINT);

	// Runtime Tracking Point - The current task is resuming execution
	TrackingPoints::exitUnblockTask(task, currentTask, fromUserCode);
}

uint64_t BlockingAPI::waitForUs(uint64_t timeUs)
{
	if (timeUs == 0)
		return 0;

	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);

	CPU *cpu = currentThread->getComputePlace();
	Task *currentTask = currentThread->getTask();
	assert(cpu != nullptr);
	assert(currentTask != nullptr);

	// Runtime Tracking Point - The current task is gonna be readded to the scheduler
	TrackingPoints::enterWaitFor(currentTask);

	Task::deadline_t timeout = (Task::deadline_t) timeUs;

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

	// Re-add the current task to the scheduler with a deadline
	Scheduler::addReadyTask(currentTask, cpu, DEADLINE_TASK_HINT);

	TaskBlocking::taskBlocks(currentThread, currentTask);

	// Update the CPU since the thread may have migrated
	cpu = currentThread->getComputePlace();
	assert(cpu != nullptr);
	Instrument::ThreadInstrumentationContext::updateComputePlace(cpu->getInstrumentationId());

	// Runtime Tracking Point - The current task is resuming execution
	TrackingPoints::exitWaitFor(currentTask);

	return (uint64_t) (Chrono::now<Task::deadline_t>() - start);
}

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
	BlockingAPI::blockCurrentTask(true);
}

extern "C" void nanos6_unblock_task(void *blocking_context)
{
	Task *task = static_cast<Task *>(blocking_context);
	BlockingAPI::unblockTask(task, true);
}

extern "C" uint64_t nanos6_wait_for(uint64_t time_us)
{
	return BlockingAPI::waitForUs(time_us);
}
