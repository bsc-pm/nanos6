/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef LOOP_GENERATOR_HPP
#define LOOP_GENERATOR_HPP

#include "Taskfor.hpp"
#include "Taskloop.hpp"
#include "system/ompss/AddTask.hpp"

#include <InstrumentAddTask.hpp>
#include <InstrumentTaskId.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>


class LoopGenerator {
public:
	static inline Taskfor *createTaskforCollaborator(Taskfor *parent, Taskfor::bounds_t &bounds, ComputePlace *computePlace)
	{
		assert(parent != nullptr);

		nanos6_task_info_t *parentTaskInfo = parent->getTaskInfo();
		nanos6_task_invocation_info_t *parentTaskInvocationInfo = parent->getTaskInvokationInfo();
		Instrument::task_id_t parentTaskInstrumentationId = parent->getInstrumentationTaskId();

		void *originalArgsBlock = parent->getArgsBlock();
		size_t originalArgsBlockSize = parent->getArgsBlockSize();

		Taskfor *taskfor = nullptr;

		void *taskfor_ptr = (void *) computePlace->getPreallocatedTaskfor();
		taskfor = (Taskfor *) taskfor_ptr;
		void *argsBlock = nullptr;
		bool hasPreallocatedArgsBlock = parent->hasPreallocatedArgsBlock();
		if (hasPreallocatedArgsBlock) {
			assert(parentTaskInfo->duplicate_args_block != nullptr);
			parentTaskInfo->duplicate_args_block(originalArgsBlock, &argsBlock);
		} else {
			argsBlock = computePlace->getPreallocatedArgsBlock(originalArgsBlockSize);
		}

		nanos6_create_preallocated_task(parentTaskInfo, parentTaskInvocationInfo, parentTaskInstrumentationId, originalArgsBlockSize, (void *) argsBlock, taskfor_ptr, parent->getFlags());
		assert(argsBlock != nullptr);
		assert(taskfor_ptr != nullptr);

		// Copy the args block if it was not duplicated
		if (!hasPreallocatedArgsBlock) {
			assert(!parent->hasPreallocatedArgsBlock());
			memcpy(argsBlock, originalArgsBlock, originalArgsBlockSize);
		}

		// Set the flags
		taskfor->setRunnable(true);
		taskfor->setDelayedRelease(false);

		// Set the parent
		taskfor->setParent(parent);

		// Set the bounds
		taskfor->setBounds(bounds);

		// Instrument the task creation
		Instrument::task_id_t taskInstrumentationId = taskfor->getInstrumentationTaskId();
		Instrument::exitAddTaskforCollaborator(parentTaskInstrumentationId, taskInstrumentationId);

		return taskfor;
	}

	static inline void createTaskloopExecutor(nanos6_task_info_t *taskInfo,
											  nanos6_task_invocation_info_t *taskInvocationInfo,
											  size_t const &originalArgsBlockSize,
											  void const *originalArgsBlock,
											  size_t const &flags,
											  bool const &preallocatedArgsBlock,
											  Taskloop::bounds_t &bounds)
	{
		void *argsBlock;
		Taskloop *taskloop = nullptr;

		nanos6_create_task(taskInfo, taskInvocationInfo, originalArgsBlockSize, &argsBlock, (void **)&taskloop, flags, 0);
		assert(argsBlock != nullptr);
		assert(taskloop != nullptr);

		// Copy the args block
		if (preallocatedArgsBlock) {
			assert(taskInfo->duplicate_args_block != nullptr);
			taskInfo->duplicate_args_block(originalArgsBlock, &argsBlock);
		} else {
			memcpy(argsBlock, originalArgsBlock, originalArgsBlockSize);
		}

		// Set bounds of grainsize
		Taskloop::bounds_t &childBounds = taskloop->getBounds();
		Taskloop::bounds_t &myBounds = bounds;
		childBounds.lower_bound = myBounds.lower_bound;
		myBounds.lower_bound = std::min(myBounds.lower_bound + myBounds.grainsize, myBounds.upper_bound);
		childBounds.upper_bound = myBounds.lower_bound;

		// Register deps
		nanos6_submit_task((void *)taskloop);
	}
};

#endif // LOOP_GENERATOR_HPP
