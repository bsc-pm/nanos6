/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASKLOOP_GENERATOR_HPP
#define TASKLOOP_GENERATOR_HPP

#include <nanos6.h>
#include "Taskloop.hpp"
#include "TaskloopInfo.hpp"

#include <InstrumentAddTask.hpp>
#include <InstrumentTaskStatus.hpp>
#include <InstrumentThreadInstrumentationContextImplementation.hpp>

class TaskloopGenerator {
private:
	typedef nanos6_taskloop_bounds_t bounds_t;
	
public:
	static inline Taskloop* createCollaborator(Taskloop *parent)
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
		
		// Set the flags
		taskloop->setRunnable(true);
		taskloop->setDelayedDataAccessRelease(false);
		
		// Complete the taskloop creation
		completeCreation(taskloop, parent);
		
		return taskloop;
	}
	
	static inline Taskloop* createPartition(Taskloop *parent, const bounds_t &assignedBounds)
	{
		assert(parent != nullptr);
		
		void *argsBlock = nullptr;
		Taskloop *taskloop = nullptr;
		bounds_t *taskloopBounds = nullptr;
		
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
		
		// Assign the corresponding iterations
		TaskloopInfo &taskloopInfo = taskloop->getTaskloopInfo();
		taskloopInfo.initialize(assignedBounds);
		
		// Complete the taskloop creation
		completeCreation(taskloop, parent);
		
		return taskloop;
	}
	
private:
	static inline void completeCreation(Taskloop *taskloop, Taskloop *parent)
	{
		assert(taskloop != nullptr);
		assert(parent != nullptr);
		
		// Set the parent
		taskloop->setParent(parent);
		
		// Instrument the task creation
		Instrument::task_id_t taskInstrumentationId = taskloop->getInstrumentationTaskId();
		Instrument::createdTask(taskloop, taskInstrumentationId);
		Instrument::exitAddTask(taskInstrumentationId);
	}
};

#endif // TASKLOOP_GENERATOR_HPP
