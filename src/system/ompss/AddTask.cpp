/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

// This is for posix_memalign
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include <nanos6.h>
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "MemoryAllocator.hpp"
#include "scheduling/Scheduler.hpp"
#include "system/If0Task.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"
#include "tasks/Taskloop.hpp"
#include "tasks/TaskloopInfo.hpp"

#include <DataAccessRegistration.hpp>
#include <HardwareCounters.hpp>
#include <InstrumentAddTask.hpp>
#include <InstrumentTaskStatus.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <Monitoring.hpp>

#include <cassert>
#include <cstdlib>


#define DATA_ALIGNMENT_SIZE sizeof(void *)
#define TASK_ALIGNMENT 128

void nanos6_create_task(
	nanos6_task_info_t *taskInfo,
	nanos6_task_invocation_info_t *taskInvocationInfo,
	size_t args_block_size,
	void **args_block_pointer,
	void **task_pointer,
	size_t flags,
	__attribute__((unused)) size_t num_deps
) {
	assert(taskInfo->implementation_count == 1); //TODO: Temporary check until multiple implementations are supported
	
	Instrument::task_id_t taskId = Instrument::enterAddTask(taskInfo, taskInvocationInfo, flags);
	
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	if (currentWorkerThread != nullptr) {
		Task *parent = currentWorkerThread->getTask();
		if (parent != nullptr) {
			Monitoring::taskChangedStatus(parent, runtime_status, currentWorkerThread->getComputePlace());
			HardwareCounters::stopTaskMonitoring(parent);
		}
	}
	
	// Operate directly over references to the user side variables
	void *&args_block = *args_block_pointer;
	void *&task = *task_pointer;
	
	bool isTaskloop = flags & nanos6_task_flag_t::nanos6_taskloop_task;
	size_t originalArgsBlockSize = args_block_size;
	size_t taskSize = (isTaskloop) ? sizeof(Taskloop) : sizeof(Task);
	
	bool hasPreallocatedArgsBlock = (flags & nanos6_preallocated_args_block);
	
	if (hasPreallocatedArgsBlock) {
		assert(args_block != nullptr);
		task = MemoryAllocator::alloc(taskSize);
	} else {
		// Alignment fixup
		size_t missalignment = args_block_size & (DATA_ALIGNMENT_SIZE - 1);
		size_t correction = (DATA_ALIGNMENT_SIZE - missalignment) & (DATA_ALIGNMENT_SIZE - 1);
		args_block_size += correction;
		
		// Allocation and layout
		*args_block_pointer = MemoryAllocator::alloc(args_block_size + taskSize);
		
		task = (char *)args_block + args_block_size;
	}
	
	if (isTaskloop) {
		new (task) Taskloop(args_block, originalArgsBlockSize, taskInfo, taskInvocationInfo, nullptr, taskId, flags);
	} else {
		// Construct the Task object
		new (task) Task(args_block, originalArgsBlockSize, taskInfo, taskInvocationInfo, /* Delayed to the submit call */ nullptr, taskId, flags);
	}
	
}


void nanos6_submit_task(void *taskHandle)
{
	Task *task = (Task *) taskHandle;
	assert(task != nullptr);
	
	Instrument::task_id_t taskInstrumentationId = task->getInstrumentationTaskId();
	
	Task *parent = nullptr;
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	ComputePlace *computePlace = nullptr;
	
	//! A WorkerThread might spawn a remote task through a polling service,
	//! i.e. while not executing a Task already. So here, we need to check
	//! both, if we are running from inside a WorkerThread as well as if
	//! we are running a Task
	if (currentWorkerThread != nullptr && currentWorkerThread->getTask() != nullptr) {
		parent = currentWorkerThread->getTask();
		assert(parent != nullptr);
		
		computePlace = currentWorkerThread->getComputePlace();
		assert(computePlace != nullptr);
		
		task->setParent(parent);
	}
	
	Instrument::createdTask(task, taskInstrumentationId);
	
	HardwareCounters::taskCreated(task);
	Monitoring::taskCreated(task);
	
	bool ready = true;
	nanos6_task_info_t *taskInfo = task->getTaskInfo();
	assert(taskInfo != 0);
	if (taskInfo->register_depinfo != 0) {
		Instrument::ThreadInstrumentationContext instrumentationContext(taskInstrumentationId);
		ready = DataAccessRegistration::registerTaskDataAccesses(task, computePlace, computePlace->getDependencyData());
	}
	
	bool isIf0 = task->isIf0();
	
	if (ready && !isIf0) {
		// Queue the task if ready but not if0
		SchedulerInterface::ReadyTaskHint schedulingHint = SchedulerInterface::NO_HINT;
		
		if (currentWorkerThread != nullptr) {
			schedulingHint = SchedulerInterface::CHILD_TASK_HINT;
		}
		
		ComputePlace *idleComputePlace = Scheduler::addReadyTask(task, computePlace, schedulingHint);
		
		if (idleComputePlace != nullptr) {
			ThreadManager::resumeIdle((CPU *) idleComputePlace);
		}
	} else if (!ready) {
		Instrument::taskIsPending(taskInstrumentationId);
		
		Monitoring::taskChangedStatus(task, pending_status);
		HardwareCounters::stopTaskMonitoring(task);
	}
	
	if (parent != nullptr) {
		HardwareCounters::startTaskMonitoring(parent);
		Monitoring::taskChangedStatus(parent, executing_status, currentWorkerThread->getComputePlace());
	}
	
	Instrument::exitAddTask(taskInstrumentationId);
	
	// Special handling for if0 tasks
	if (isIf0) {
		if (ready) {
			// Ready if0 tasks are executed inline
			If0Task::executeInline(currentWorkerThread, parent, task, computePlace);
		} else {
			// Non-ready if0 tasks cause this thread to get blocked
			If0Task::waitForIf0Task(currentWorkerThread, parent, task, computePlace);
		}
	}
}

