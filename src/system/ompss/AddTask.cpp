/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

// This is for posix_memalign
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include <cassert>
#include <cstdlib>

#include <nanos6.h>

#include "AddTask.hpp"
#include "MemoryAllocator.hpp"
#include "executors/threads/CPUManager.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/HardwareInfo.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "scheduling/Scheduler.hpp"
#include "system/If0Task.hpp"
#include "tasks/StreamExecutor.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"
#include "tasks/Taskfor.hpp"
#include "tasks/TaskforInfo.hpp"

#include <DataAccessRegistration.hpp>
#include <HardwareCounters.hpp>
#include <InstrumentAddTask.hpp>
#include <InstrumentTaskStatus.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <Monitoring.hpp>


#define DATA_ALIGNMENT_SIZE sizeof(void *)
#define TASK_ALIGNMENT 128

void nanos6_create_task(
	nanos6_task_info_t *taskInfo,
	nanos6_task_invocation_info_t *taskInvocationInfo,
	size_t args_block_size,
	void **args_block_pointer,
	void **task_pointer,
	size_t flags,
	size_t num_deps
) {
	assert(taskInfo->implementation_count == 1); //TODO: Temporary check until multiple implementations are supported

	nanos6_device_t taskDeviceType = (nanos6_device_t) taskInfo->implementations[0].device_type_id;
	if (!HardwareInfo::canDeviceRunTasks(taskDeviceType)) {
		FatalErrorHandler::failIf(true, "Task of device type '", taskDeviceType, "' has no active hardware associated");
	}

	Instrument::task_id_t taskId = Instrument::enterAddTask(taskInfo, taskInvocationInfo, flags);

	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	if (currentWorkerThread != nullptr) {
		Task *parent = currentWorkerThread->getTask();
		if (parent != nullptr) {
			Monitoring::taskChangedStatus(parent, runtime_status);
			HardwareCounters::stopTaskMonitoring(parent);
		}
	}

	// Operate directly over references to the user side variables
	void *&args_block = *args_block_pointer;
	void *&task = *task_pointer;

	bool isTaskfor = flags & nanos6_task_flag_t::nanos6_taskloop_task;
	//bool isTaskfor = flags & nanos6_task_flag_t::nanos6_taskfor_task;
	bool isStreamExecutor = flags & (1 << Task::stream_executor_flag);
	size_t originalArgsBlockSize = args_block_size;
	size_t taskSize;
	if (isTaskfor) {
		taskSize = sizeof(Taskfor);
	} else if (isStreamExecutor) {
		taskSize = sizeof(StreamExecutor);
	} else {
		taskSize = sizeof(Task);
	}

#ifdef DISCRETE_DEPS
	// We use num_deps to create the correctly sized array for storing the dependencies.
	// Two plain C arrays are used, one for the actual DataAccess structures and another for the
	// addresses, which is the one used for searching. That way we cause less cache misses searching.

	size_t seqsSize = sizeof(DataAccess) * num_deps;
	size_t addrSize = sizeof(void *) * num_deps;
#else
	size_t seqsSize = 0;
	size_t addrSize = 0;
#endif

	bool hasPreallocatedArgsBlock = (flags & nanos6_preallocated_args_block);

	if (hasPreallocatedArgsBlock) {
		assert(args_block != nullptr);
		assert(seqsSize == 0 && addrSize == 0);
		task = MemoryAllocator::alloc(taskSize);
	} else {
		// Alignment fixup
		size_t missalignment = args_block_size & (DATA_ALIGNMENT_SIZE - 1);
		size_t correction = (DATA_ALIGNMENT_SIZE - missalignment) & (DATA_ALIGNMENT_SIZE - 1);
		args_block_size += correction;

		// Allocation and layout
		*args_block_pointer = MemoryAllocator::alloc(args_block_size + taskSize + seqsSize + addrSize);
		task = (char *)args_block + args_block_size;
	}

	Instrument::createdArgsBlock(taskId, *args_block_pointer, originalArgsBlockSize, args_block_size);

	void * seqs = (char *)task + taskSize;
	void * addresses = (char *)seqs + seqsSize;

	if (isTaskfor) {
		// Taskfor is always final.
		flags |= nanos6_task_flag_t::nanos6_final_task;
		new (task) Taskfor (args_block, originalArgsBlockSize, taskInfo, taskInvocationInfo, nullptr, taskId, flags);
	} else if (isStreamExecutor) {
		new (task) StreamExecutor(args_block, originalArgsBlockSize, taskInfo, taskInvocationInfo, nullptr, taskId, flags);
	} else {
		// Construct the Task object
		new (task) Task(args_block, originalArgsBlockSize, taskInfo, taskInvocationInfo, /* Delayed to the submit call */ nullptr, taskId, flags, seqs, addresses, num_deps);
	}

}

void nanos6_create_preallocated_task(
	nanos6_task_info_t *taskInfo,
	nanos6_task_invocation_info_t *taskInvocationInfo,
	Instrument::task_id_t parentTaskInstrumentationId,
	size_t args_block_size,
	void *preallocatedArgsBlock,
	void *preallocatedTask,
	size_t flags
) {
	assert(taskInfo->implementation_count == 1); //TODO: Temporary check until multiple implementations are supported
	assert(preallocatedArgsBlock != nullptr);
	assert(preallocatedTask != nullptr);

	Instrument::task_id_t taskId = Instrument::enterAddTaskforCollaborator(parentTaskInstrumentationId, taskInfo, taskInvocationInfo, flags);

	bool isTaskfor = flags & nanos6_task_flag_t::nanos6_taskloop_task;
	//bool isTaskfor = flags & nanos6_task_flag_t::nanos6_taskfor_task;
	FatalErrorHandler::failIf(!isTaskfor, "Only taskfors can be created this way.");

	Taskfor *taskfor = (Taskfor *) preallocatedTask;
	taskfor->reinitialize(preallocatedArgsBlock, args_block_size, taskInfo, taskInvocationInfo, nullptr, taskId, flags);
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

	Instrument::createdTask(task, taskInstrumentationId);

	HardwareCounters::taskCreated(task);
	Monitoring::taskCreated(task);

	bool ready = true;
	nanos6_task_info_t *taskInfo = task->getTaskInfo();
	assert(taskInfo != 0);
	if (taskInfo->register_depinfo != 0) {
		// Begin as pending status, become ready later, through the scheduler
		Instrument::ThreadInstrumentationContext instrumentationContext(taskInstrumentationId);
		Instrument::taskIsPending(taskInstrumentationId);

		Monitoring::taskChangedStatus(task, pending_status);
		// No need to stop hardware counters, as the task was created just now

		ready = DataAccessRegistration::registerTaskDataAccesses(task, computePlace, computePlace->getDependencyData());
	}
	assert(parent != nullptr || ready);

	bool isIf0 = task->isIf0();

	// We cannot execute inline tasks that are not to run in the host
	bool executesInDevice = (task->getDeviceType() != nanos6_host_device);

	if (ready && (!isIf0 || executesInDevice)) {
		// Queue the task if ready but not if0
		ReadyTaskHint schedulingHint = NO_HINT;

		if (currentWorkerThread != nullptr) {
			schedulingHint = CHILD_TASK_HINT;
		}

		Scheduler::addReadyTask(task, computePlace, schedulingHint);

		// After adding a task, the CPUManager may want to unidle CPUs
		CPUManager::executeCPUManagerPolicy(computePlace, ADDED_TASKS, 1);
	}

	if (parent != nullptr) {
		HardwareCounters::startTaskMonitoring(parent);
		Monitoring::taskChangedStatus(parent, executing_status);
	}

	Instrument::exitAddTask(taskInstrumentationId);

	// Special handling for if0 tasks
	if (isIf0) {
		if (ready && !executesInDevice) {
			// Ready if0 tasks are executed inline, if they are not device tasks
			If0Task::executeInline(currentWorkerThread, parent, task, computePlace);
		} else {
			// Non-ready if0 tasks cause this thread to get blocked
			If0Task::waitForIf0Task(currentWorkerThread, parent, task, computePlace);
		}
	}
}
