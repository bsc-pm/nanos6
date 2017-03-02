// This is for posix_memalign
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include <nanos6.h>
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/places/HardwarePlace.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "scheduling/Scheduler.hpp"
#include "system/If0Task.hpp"
#include "tasks/Task.hpp"

#include <DataAccessRegistration.hpp>

#include <InstrumentAddTask.hpp>
#include <InstrumentTaskStatus.hpp>

#include <cassert>
#include <cstdlib>


#define DATA_ALIGNMENT_SIZE sizeof(void *)
#define TASK_ALIGNMENT 128


void nanos_create_task(
	nanos_task_info *taskInfo,
	nanos_task_invocation_info *taskInvocationInfo,
	size_t args_block_size,
	void **args_block_pointer,
	void **task_pointer,
	size_t flags
) {
	Instrument::task_id_t taskId = Instrument::enterAddTask(taskInfo, taskInvocationInfo);
	
	// Alignment fixup
	size_t missalignment = args_block_size & (DATA_ALIGNMENT_SIZE - 1);
	size_t correction = (DATA_ALIGNMENT_SIZE - missalignment) & (DATA_ALIGNMENT_SIZE - 1);
	args_block_size += correction;
	
	// Allocation and layout
	int rc = posix_memalign(args_block_pointer, TASK_ALIGNMENT, args_block_size + sizeof(Task));
	FatalErrorHandler::handle(rc, " when trying to allocate memory for a new task of type '", taskInfo->task_label, "' with args block of size ", args_block_size);
	
	// Operate directly over references to the user side variables
	void *&args_block = *args_block_pointer;
	void *&task = *task_pointer;
	
	task = (char *)args_block + args_block_size;
	
	// Construct the Task object
	new (task) Task(args_block, taskInfo, taskInvocationInfo, /* Delayed to the submit call */ nullptr, taskId, flags);
}


void nanos_submit_task(void *taskHandle)
{
	Task *task = (Task *) taskHandle;
	assert(task != nullptr);
	
	Task *parent = nullptr;
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	HardwarePlace *hardwarePlace = nullptr;
	
	if (currentWorkerThread != nullptr) {
		assert(currentWorkerThread->getTask() != nullptr);
		parent = currentWorkerThread->getTask();
		assert(parent != nullptr);
		
		hardwarePlace = currentWorkerThread->getHardwarePlace();
		assert(hardwarePlace != nullptr);
		
		task->setParent(parent);
	}
	
	Instrument::createdTask(task, task->getInstrumentationTaskId());
	
	bool ready = true;
	nanos_task_info *taskInfo = task->getTaskInfo();
	assert(taskInfo != 0);
	if (taskInfo->register_depinfo != 0) {
		ready = DataAccessRegistration::registerTaskDataAccesses(task);
	}
	
	bool isIf0 = task->isIf0();
	
	if (ready && !isIf0) {
		// Queue the task if ready but not if0
		SchedulerInterface::ReadyTaskHint schedulingHint = SchedulerInterface::NO_HINT;
		
		if (currentWorkerThread != nullptr) {
			schedulingHint = SchedulerInterface::CHILD_TASK_HINT;
		}
		
		HardwarePlace *idleHardwarePlace = Scheduler::addReadyTask(task, hardwarePlace, schedulingHint);
		
		if (idleHardwarePlace != nullptr) {
			ThreadManager::resumeIdle((CPU *) idleHardwarePlace);
		}
	} else if (!ready) {
		Instrument::taskIsPending(task->getInstrumentationTaskId());
	}
	
	Instrument::exitAddTask(task->getInstrumentationTaskId());
	
	// Special handling for if0 tasks
	if (isIf0) {
		if (ready) {
			// Ready if0 tasks are executed inline
			If0Task::executeInline(currentWorkerThread, parent, task);
		} else {
			// Non-ready if0 tasks cause this thread to get blocked
			If0Task::waitForIf0Task(currentWorkerThread, parent, task, hardwarePlace);
		}
	}
	
	
}

