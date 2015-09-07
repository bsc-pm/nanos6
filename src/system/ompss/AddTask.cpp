// This is for posix_memalign
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include "api/nanos6_rt_interface.h"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/places/HardwarePlace.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "scheduling/Scheduler.hpp"
#include <tasks/Task.hpp>

#include <cassert>
#include <cstdlib>
#include <iostream>

#ifndef NDEBUG
#include "system/MainTask.hpp"
#endif


#define DATA_ALIGNMENT_SIZE sizeof(void *)
#define TASK_ALIGNMENT 128


void nanos_create_task(nanos_task_info *taskInfo, nanos_task_invocation_info *taskInvokationInfo, size_t args_block_size, void **args_block_pointer, void **task_pointer)
{
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
	new (task) Task(args_block, taskInfo, taskInvokationInfo, /* Delayed to the submit call */ nullptr);
}


void nanos_submit_task(void *taskHandle)
{
	Task *task = (Task *) taskHandle;
	assert(task != nullptr);
	
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	
	HardwarePlace *hardwarePlace = nullptr;
	if (__builtin_expect((currentWorkerThread != nullptr), 1)) {
		assert(currentWorkerThread->getTask() != nullptr);
		task->setParent(currentWorkerThread->getTask());
		
		hardwarePlace = currentWorkerThread->getHardwarePlace();
		assert(hardwarePlace != nullptr);
	} else {
		// Adding the main task from within the leader thread
		assert(task->getTaskInfo() == &nanos6::main_task_info);
	}
	
	HardwarePlace *idleHardwarePlace = Scheduler::addReadyTask(task, hardwarePlace);
	assert((currentWorkerThread != nullptr) || (idleHardwarePlace == nullptr)); // The main task is added before the scheduler 
	
	if (idleHardwarePlace != nullptr) {
		ThreadManager::resumeIdle((CPU *) idleHardwarePlace);
	}
}

