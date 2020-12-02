/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

// This is for posix_memalign
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include <cassert>
#include <cstdlib>

#include <nanos6.h>

#include "AddTask.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/HardwareInfo.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "hardware-counters/TaskHardwareCounters.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "monitoring/Monitoring.hpp"
#include "scheduling/Scheduler.hpp"
#include "system/If0Task.hpp"
#include "system/Throttle.hpp"
#include "system/TrackingPoints.hpp"
#include "tasks/StreamExecutor.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"
#include "tasks/Taskfor.hpp"
#include "tasks/Taskloop.hpp"

#include <DataAccessRegistration.hpp>
#include <InstrumentAddTask.hpp>
#include <InstrumentTaskStatus.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <MemoryAllocator.hpp>
#include <TaskDataAccesses.hpp>
#include <TaskDataAccessesInfo.hpp>


#define DATA_ALIGNMENT_SIZE sizeof(void *)

Task *AddTask::createTask(
	nanos6_task_info_t *taskInfo,
	nanos6_task_invocation_info_t *taskInvocationInfo,
	void *argsBlock,
	size_t argsBlockSize,
	size_t flags,
	size_t numDependencies,
	bool fromUserCode
) {
	Task *task = nullptr;
	Task *creator = nullptr;
	WorkerThread *workerThread = WorkerThread::getCurrentWorkerThread();
	if (workerThread != nullptr) {
		creator = workerThread->getTask();
	}

	// Runtime Tracking Point - Enter the creation of a task
	Instrument::task_id_t taskId = TrackingPoints::enterCreateTask(
		creator, taskInfo, taskInvocationInfo, flags, fromUserCode
	);

	// Throttle. If active, act as a taskwait
	if (Throttle::isActive() && creator != nullptr) {
		assert(workerThread != nullptr);
		// We will try to execute something else instead of creating more memory pressure
		// on the system
		while (Throttle::engage(creator, workerThread));
	}

	bool isTaskfor = flags & nanos6_taskfor_task;
	bool isTaskloop = flags & nanos6_taskloop_task;
	bool isTaskloopFor = (isTaskloop && isTaskfor);
	bool isStreamExecutor = flags & (1 << Task::stream_executor_flag);
	size_t originalArgsBlockSize = argsBlockSize;
	size_t taskSize;

	// A taskloop for construct enables both taskloop and taskfor flags, but we must
	// create a taskloop. Notice we first check the taskloop condition
	if (isTaskloop || isTaskloopFor) {
		taskSize = sizeof(Taskloop);
	} else if (isTaskfor) {
		taskSize = sizeof(Taskfor);
	} else if (isStreamExecutor) {
		taskSize = sizeof(StreamExecutor);
	} else {
		taskSize = sizeof(Task);
	}

	TaskDataAccessesInfo taskAccesses(numDependencies);
	size_t taskAccessesSize = taskAccesses.getAllocationSize();
	size_t taskCountersSize = TaskHardwareCounters::getAllocationSize();
	size_t taskStatisticsSize = Monitoring::getAllocationSize();

	bool hasPreallocatedArgsBlock = (flags & nanos6_preallocated_args_block);
	if (hasPreallocatedArgsBlock) {
		assert(argsBlock != nullptr);
		task = (Task *) MemoryAllocator::alloc(taskSize
			+ taskAccessesSize
			+ taskCountersSize
			+ taskStatisticsSize);
	} else {
		// Alignment fixup
		size_t missalignment = argsBlockSize & (DATA_ALIGNMENT_SIZE - 1);
		size_t correction = (DATA_ALIGNMENT_SIZE - missalignment) & (DATA_ALIGNMENT_SIZE - 1);
		argsBlockSize += correction;

		// Allocation and layout
		argsBlock = MemoryAllocator::alloc(argsBlockSize + taskSize
			+ taskAccessesSize
			+ taskCountersSize
			+ taskStatisticsSize);
		task = (Task *) ((char *) argsBlock + argsBlockSize);
	}

	Instrument::createdArgsBlock(taskId, argsBlock, originalArgsBlockSize, argsBlockSize);

	taskAccesses.setAllocationAddress((char *) task + taskSize);

	void *taskCountersAddress = (taskCountersSize > 0) ?
		(char *) task + taskSize + taskAccessesSize : nullptr;

	void *taskStatisticsAddress = (taskStatisticsSize > 0) ?
		(char *) task + taskSize + taskAccessesSize + taskCountersSize : nullptr;

	if (isTaskloop || isTaskloopFor) {
		new (task) Taskloop(argsBlock, originalArgsBlockSize,
			taskInfo, taskInvocationInfo, nullptr, taskId,
			flags, taskAccesses, taskCountersAddress, taskStatisticsAddress);
	} else if (isTaskfor) {
		// Taskfors are always final
		flags |= nanos6_final_task;

		new (task) Taskfor(argsBlock, originalArgsBlockSize,
			taskInfo, taskInvocationInfo, nullptr, taskId,
			flags, taskAccesses, taskCountersAddress, taskStatisticsAddress);
	} else if (isStreamExecutor) {
		new (task) StreamExecutor(argsBlock, originalArgsBlockSize,
			taskInfo, taskInvocationInfo, nullptr, taskId, flags,
			taskAccesses, taskCountersAddress, taskStatisticsAddress);
	} else {
		new (task) Task(argsBlock, originalArgsBlockSize,
			taskInfo, taskInvocationInfo, nullptr, taskId,
			flags, taskAccesses, taskCountersAddress, taskStatisticsAddress);
	}

	TrackingPoints::exitCreateTask(creator, fromUserCode);

	return task;
}

void AddTask::submitTask(Task *task, Task *parent, bool fromUserCode)
{
	assert(task != nullptr);

	// Retrieve the current thread, compute place, and the creator if it exists
	Task *creator = nullptr;
	ComputePlace *computePlace = nullptr;
	WorkerThread *workerThread = WorkerThread::getCurrentWorkerThread();
	if (workerThread != nullptr) {
		creator = workerThread->getTask();
		computePlace = workerThread->getComputePlace();
		assert(computePlace != nullptr);
	}

	// Set the parent and check if it is a stream executor
	if (parent != nullptr) {
		task->setParent(parent);

		if (parent->isStreamExecutor()) {
			// Check if we need to save the spawned function's id for a future
			// trigger of a callback (spawned stream functions)
			StreamExecutor *executor = (StreamExecutor *) parent;
			StreamFunctionCallback *callback = executor->getCurrentFunctionCallback();
			if (callback != nullptr) {
				task->setParentSpawnCallback(callback);
				executor->increaseCallbackParticipants(callback);
			}
		}
	}

	// Runtime Tracking Point - Enter the submission of a task to the scheduler
	TrackingPoints::enterSubmitTask(creator, task, fromUserCode);

	// Compute the task priority only when the scheduler is
	// considering the task priorities
	if (Scheduler::isPriorityEnabled()) {
		if (task->computePriority()) {
			Instrument::taskHasNewPriority(
				task->getInstrumentationTaskId(),
				task->getPriority());
		}
	}

	bool ready = true;
	nanos6_task_info_t *taskInfo = task->getTaskInfo();
	assert(taskInfo != 0);
	if (taskInfo->register_depinfo != 0) {
		assert(computePlace != nullptr);

		Instrument::task_id_t taskInstrumentationId = task->getInstrumentationTaskId();
		Instrument::ThreadInstrumentationContext instrumentationContext(taskInstrumentationId);

		// Runtime Tracking Point - The created task has unresolved dependencies and is pending
		TrackingPoints::taskIsPending(task);

		ready = DataAccessRegistration::registerTaskDataAccesses(task, computePlace, computePlace->getDependencyData());
	}

	bool executesInDevice = (task->getDeviceType() != nanos6_host_device);
	bool isIf0 = task->isIf0();

	assert(parent != nullptr || ready);
	assert(parent != nullptr || !isIf0);

	if (ready && (!isIf0 || executesInDevice)) {
		// Queue the task if ready and not if0. Device if0 ready tasks must be
		// queued too; they are managed by the device scheduling infrastructure
		ReadyTaskHint hint = (parent != nullptr) ? CHILD_TASK_HINT : NO_HINT;

		Scheduler::addReadyTask(task, computePlace, hint);
	}

	// Special handling for if0 tasks
	if (isIf0) {
		if (ready && !executesInDevice) {
			// Ready if0 tasks are executed inline, if they are not device tasks
			If0Task::executeInline(workerThread, parent, task, computePlace);
		} else {
			// Non-ready if0 tasks cause this thread to get blocked
			If0Task::waitForIf0Task(workerThread, parent, task, computePlace);
		}
	}

	// Runtime Tracking Point - Exit the submission of a task (and thus, the creation)
	TrackingPoints::exitSubmitTask(creator, task, fromUserCode);
}


//! Public API function to create tasks
void nanos6_create_task(
	nanos6_task_info_t *task_info,
	nanos6_task_invocation_info_t *task_invocation_info,
	size_t args_block_size,
	void **args_block_pointer,
	void **task_pointer,
	size_t flags,
	size_t num_deps
) {
	// TODO: Temporary check until multiple implementations are supported
	assert(task_info->implementation_count == 1);

	nanos6_device_t deviceType = (nanos6_device_t) task_info->implementations[0].device_type_id;
	if (!HardwareInfo::canDeviceRunTasks(deviceType)) {
		FatalErrorHandler::fail("No hardware associated for task device type", deviceType);
	}

	Task *task = AddTask::createTask(
		task_info, task_invocation_info,
		*args_block_pointer, args_block_size,
		flags, num_deps, true
	);
	assert(task != nullptr);

	*task_pointer = (void *) task;
	*args_block_pointer = task->getArgsBlock();
}

//! Public API function to submit tasks
void nanos6_submit_task(void *task_handle)
{
	Task *task = (Task *) task_handle;
	assert(task != nullptr);

	WorkerThread *workerThread = WorkerThread::getCurrentWorkerThread();
	assert(workerThread != nullptr);

	Task *parent = workerThread->getTask();
	assert(parent != nullptr);

	AddTask::submitTask(task, parent, true);
}
