/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include <nanos6/blocking.h>

#include "DataAccessRegistration.hpp"
#include "ompss/TaskBlocking.hpp"

#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/ThreadManagerPolicy.hpp"
#include "executors/threads/WorkerThread.hpp"

#include "scheduling/Scheduler.hpp"

#include <InstrumentBlocking.hpp>


extern "C" void *nanos_get_current_blocking_context()
{
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);
	
	Task *currentTask = currentThread->getTask();
	assert(currentTask != nullptr);
	
	return currentTask;
}


extern "C" void nanos_block_current_task(__attribute__((unused)) void *blocking_context)
{
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);
	
	__attribute__((unused)) CPU *cpu = nullptr;
	cpu = currentThread->getComputePlace();
	assert(cpu != nullptr);
	
	Task *currentTask = currentThread->getTask();
	assert(currentTask != nullptr);
	
	assert(blocking_context == currentTask);
	
	Instrument::taskIsBlocked(currentTask->getInstrumentationTaskId(), Instrument::user_requested_blocking_reason);
	Instrument::enterBlocking(currentTask->getInstrumentationTaskId());
	
	DataAccessRegistration::handleEnterBlocking(currentTask);
	TaskBlocking::taskBlocks(currentThread, currentTask, ThreadManagerPolicy::POLICY_NO_INLINE);
	DataAccessRegistration::handleExitBlocking(currentTask);
	
	Instrument::exitBlocking(currentTask->getInstrumentationTaskId());
	Instrument::taskIsExecuting(currentTask->getInstrumentationTaskId());
}


extern "C" void nanos_unblock_task(void *blocking_context)
{
	Task *task = static_cast<Task *>(blocking_context);
	
	Instrument::unblockTask(task->getInstrumentationTaskId());
	Scheduler::addReadyTask(task, nullptr, SchedulerInterface::UNBLOCKED_TASK_HINT);
	
	CPU *idleCPU = (CPU *) Scheduler::getIdleComputePlace();
	if (idleCPU != nullptr) {
		ThreadManager::resumeIdle(idleCPU);
	}
}

