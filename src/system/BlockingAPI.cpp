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
#include "ompss/TaskBlocking.hpp"
#include "scheduling/Scheduler.hpp"
#include "support/chronometers/std/Chrono.hpp"
#include "system/ompss/MetricPoints.hpp"


void BlockingAPI::blockCurrentTask(bool fromUserCode)
{
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);

	Task *currentTask = currentThread->getTask();
	assert(currentTask != nullptr);

	// Runtime Core Metric Point - The current task is gonna be blocked
	Instrument::task_id_t taskId = currentTask->getInstrumentationTaskId();
	MetricPoints::enterBlockCurrentTask(currentTask, taskId, fromUserCode);

	TaskBlocking::taskBlocks(currentThread, currentTask);

	// Update the CPU as this thread may have migrated
	ComputePlace *computePlace = currentThread->getComputePlace();
	assert(computePlace != nullptr);
	Instrument::ThreadInstrumentationContext::updateComputePlace(computePlace->getInstrumentationId());

	// Runtime Core Metric Point - The current task resumes its execution
	MetricPoints::exitBlockCurrentTask(currentTask, taskId, fromUserCode);
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

	// See taskRuntimeTransition variable note in spawnFunction() for more details
	bool taskRuntimeTransition = fromUserCode && (currentTask != nullptr);
	Instrument::task_id_t taskId = task->getInstrumentationTaskId();

	// Runtime Core Metric Point - The current task is gonna execute runtime code
	MetricPoints::enterUnblockCurrentTask(currentTask, taskId, taskRuntimeTransition);

	Scheduler::addReadyTask(task, computePlace, UNBLOCKED_TASK_HINT);

	// Runtime Core Metric Point - The current task is resuming execution
	MetricPoints::exitUnblockCurrentTask(currentTask, taskId, taskRuntimeTransition);
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
	if (time_us == 0)
		return 0;

	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);

	CPU *cpu = currentThread->getComputePlace();
	Task *currentTask = currentThread->getTask();
	assert(cpu != nullptr);
	assert(currentTask != nullptr);

	// Runtime Core Metric Point - The current task is gonna be readded to the scheduler
	Instrument::task_id_t taskId = currentTask->getInstrumentationTaskId();
	MetricPoints::enterWaitFor(currentTask, taskId);

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

	// Re-add the current task to the scheduler with a deadline
	Scheduler::addReadyTask(currentTask, cpu, DEADLINE_TASK_HINT);

	TaskBlocking::taskBlocks(currentThread, currentTask);

	// Update the CPU since the thread may have migrated
	cpu = currentThread->getComputePlace();
	assert(cpu != nullptr);
	Instrument::ThreadInstrumentationContext::updateComputePlace(cpu->getInstrumentationId());

	// Runtime Core Metric Point - The current task is resuming execution
	MetricPoints::exitWaitFor(currentTask, taskId);

	return (uint64_t) (Chrono::now<Task::deadline_t>() - start);
}
