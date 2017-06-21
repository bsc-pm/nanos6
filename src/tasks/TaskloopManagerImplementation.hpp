#ifndef TASKLOOP_MANAGER_IMPLEMENTATION_HPP
#define TASKLOOP_MANAGER_IMPLEMENTATION_HPP

#include <nanos6.h>
#include "Task.hpp"
#include "Taskloop.hpp"
#include "TaskloopInfo.hpp"
#include "TaskloopManager.hpp"
#include "executors/threads/WorkerThread.hpp"

#include <InstrumentAddTask.hpp>
#include <InstrumentTaskStatus.hpp>
#include <InstrumentThreadInstrumentationContextImplementation.hpp>


inline Taskloop* TaskloopManager::createRunnableTaskloop(Taskloop *parent, const nanos6_taskloop_bounds_t &assignedBounds)
{
	assert(parent != nullptr);
	
	size_t flags = parent->getFlags();
	nanos_task_info *taskInfo = parent->getTaskInfo();
	nanos_task_invocation_info *taskInvocationInfo = parent->getTaskInvokationInfo();
	
	void *originalArgsBlock = parent->getArgsBlock();
	size_t originalArgsBlockSize = parent->getArgsBlockSize();
	
	Taskloop *taskloop = nullptr;
	void *argsBlock = nullptr;
	void *bounds = nullptr;
	
	// Create the task for this partition
	nanos_create_task(taskInfo, taskInvocationInfo, originalArgsBlockSize, (void **) &argsBlock, (void **) &bounds, (void **) &taskloop, flags);
	assert(argsBlock != nullptr);
	assert(bounds != nullptr);
	assert(taskloop != nullptr);
	
	// Copy the args block
	memcpy(argsBlock, originalArgsBlock, originalArgsBlockSize);
	
	// Since a taskloop is non-runnable by default, set it as runnable
	taskloop->setRunnable(true);
	
	// Complete the taskloop creation
	completeTaskloopCreation(taskloop, parent, assignedBounds);
	
	return taskloop;
}

inline Taskloop* TaskloopManager::createPartitionTaskloop(Taskloop *parent, const nanos6_taskloop_bounds_t &assignedBounds)
{
	assert(parent != nullptr);
	
	void *argsBlock = nullptr;
	Taskloop *taskloop = nullptr;
	nanos6_taskloop_bounds_t *taskloopBounds = nullptr;
	
	// Get the infomation of the complete taskloop
	nanos_task_info *taskInfo = parent->getTaskInfo();
	nanos_task_invocation_info *taskInvocationInfo = parent->getTaskInvokationInfo();
	
	// Make the taskloop non-runnable
	size_t flags = parent->getFlags();
	void *originalArgsBlock = parent->getArgsBlock();
	size_t originalArgsBlockSize = parent->getArgsBlockSize();
	
	// Create the taskloop for this partition
	nanos_create_task(taskInfo, taskInvocationInfo, 0, (void **) &argsBlock, (void **) &taskloopBounds, (void **) &taskloop, flags);
	assert(argsBlock != nullptr);
	assert(taskloopBounds != nullptr);
	assert(taskloop != nullptr);
	
	// The args block point to the args block of the original taskloop
	taskloop->setArgsBlock(originalArgsBlock);
	taskloop->setArgsBlockSize(originalArgsBlockSize);
	taskloop->setArgsBlockOwner(false);
	
	// Complete the taskloop creation
	completeTaskloopCreation(taskloop, parent, assignedBounds);
	
	return taskloop;
}

void TaskloopManager::completeTaskloopCreation(Taskloop *taskloop, Taskloop *parent, const nanos6_taskloop_bounds_t &assignedBounds)
{
	assert(taskloop != nullptr);
	assert(parent != nullptr);
	
	// Assign the corresponding iterations
	TaskloopInfo &taskloopInfo = taskloop->getTaskloopInfo();
	taskloopInfo.setBounds(assignedBounds);
	
	// Set the parent
	taskloop->setParent(parent);
	
	// Instrument the task creation
	Instrument::task_id_t taskInstrumentationId = taskloop->getInstrumentationTaskId();
	Instrument::createdTask(taskloop, taskInstrumentationId);
	Instrument::exitAddTask(taskInstrumentationId);
}

#endif // TASKLOOP_MANAGER_IMPLEMENTATION_HPP
