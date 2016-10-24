// This is for posix_memalign
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 600
#endif

#include "api/nanos6_rt_interface.h"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/Task.hpp"

#include <DataAccessRegistration.hpp>

#include <InstrumentAddTask.hpp>
#include <InstrumentTaskStatus.hpp>

#include <cassert>
#include <cstdlib>


#define DATA_ALIGNMENT_SIZE sizeof(void *)
#define TASK_ALIGNMENT 128


void nanos_create_task(nanos_task_info *taskInfo, nanos_task_invocation_info *taskInvocationInfo, size_t args_block_size, void **args_block_pointer, void **task_pointer)
{
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
	new (task) Task(args_block, taskInfo, taskInvocationInfo, /* Delayed to the submit call */ nullptr, taskId);
}


void nanos_submit_task(void *taskHandle)
{
	Task *task = (Task *) taskHandle;
	assert(task != nullptr);
	
	Task *parent = nullptr;
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	ComputePlace *hardwarePlace = nullptr;
	
	if (currentWorkerThread != nullptr) {
		assert(currentWorkerThread->getTask() != nullptr);
		parent = currentWorkerThread->getTask();
		assert(parent != nullptr);
		
		hardwarePlace = currentWorkerThread->getComputePlace();
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
	
	if (ready) {
		//ComputePlace *idleComputePlace = Scheduler::addReadyTask(task, hardwarePlace, SchedulerInterface::SchedulerInterface::CHILD_TASK_HINT);
		ComputePlace *idleComputePlace = Scheduler::addPreReadyTask(task, hardwarePlace, SchedulerInterface::SchedulerInterface::CHILD_TASK_HINT);
		assert((currentWorkerThread != nullptr) || (idleComputePlace == nullptr)); // The main task is added before the scheduler
		
		if (idleComputePlace != nullptr) {
			ThreadManager::resumeIdle((CPU *) idleComputePlace);
		}
	} else {
		Instrument::taskIsPending(task->getInstrumentationTaskId());
	}
	
	Instrument::exitAddTask(task->getInstrumentationTaskId());
}

