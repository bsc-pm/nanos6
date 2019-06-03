/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include <nanos6/blocking.h>

#include "DataAccessRegistration.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/ThreadManagerPolicy.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "ompss/TaskBlocking.hpp"
#include "scheduling/Scheduler.hpp"

#include <HardwareCounters.hpp>
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
	
	ComputePlace *computePlace = currentThread->getComputePlace();
	assert(computePlace != nullptr);
	
	Task *currentTask = currentThread->getTask();
	assert(currentTask != nullptr);
	
	assert(blocking_context == currentTask);
	
	Monitoring::taskChangedStatus(currentTask, blocked_status, computePlace);
	HardwareCounters::stopTaskMonitoring(currentTask);
	
	Instrument::taskIsBlocked(currentTask->getInstrumentationTaskId(), Instrument::user_requested_blocking_reason);
	Instrument::enterBlocking(currentTask->getInstrumentationTaskId());
	
	DataAccessRegistration::handleEnterBlocking(currentTask);
	TaskBlocking::taskBlocks(currentThread, currentTask, ThreadManagerPolicy::POLICY_NO_INLINE);
	
	// Update the CPU since the thread may have migrated
	computePlace = currentThread->getComputePlace();
	assert(computePlace != nullptr);
	Instrument::ThreadInstrumentationContext::updateComputePlace(computePlace->getInstrumentationId());
	
	DataAccessRegistration::handleExitBlocking(currentTask);
	
	Instrument::exitBlocking(currentTask->getInstrumentationTaskId());
	Instrument::taskIsExecuting(currentTask->getInstrumentationTaskId());
	
	HardwareCounters::startTaskMonitoring(currentTask);
	Monitoring::taskChangedStatus(currentTask, executing_status, computePlace);
}


extern "C" void nanos6_unblock_task(void *blocking_context)
{
	Task *task = static_cast<Task *>(blocking_context);
	
	Instrument::unblockTask(task->getInstrumentationTaskId());
	Scheduler::addReadyTask(task, nullptr, SchedulerInterface::UNBLOCKED_TASK_HINT);
	
	CPU *idleCPU = (CPU *) Scheduler::getIdleComputePlace();
	if (idleCPU != nullptr) {
		ThreadManager::resumeIdle(idleCPU);
	}
}

