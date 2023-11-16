/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef LOOP_GENERATOR_HPP
#define LOOP_GENERATOR_HPP

#include "Taskloop.hpp"
#include "system/AddTask.hpp"

#include <InstrumentAddTask.hpp>
#include <InstrumentTaskId.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>


class LoopGenerator {
public:
	static inline void createTaskloopExecutor(
		Taskloop *parent,
		Taskloop::bounds_t &parentBounds,
		bool fromTaskContext = true
	) {
		assert(parent != nullptr);

		nanos6_task_info_t *parentTaskInfo = parent->getTaskInfo();
		nanos6_task_invocation_info_t *parentTaskInvocationInfo = parent->getTaskInvokationInfo();
		void *originalArgsBlock = parent->getArgsBlock();
		size_t originalArgsBlockSize = parent->getArgsBlockSize();

		void *argsBlock = nullptr;
		bool hasPreallocatedArgsBlock = parent->hasPreallocatedArgsBlock();
		if (hasPreallocatedArgsBlock) {
			assert(parentTaskInfo->duplicate_args_block != nullptr);
			parentTaskInfo->duplicate_args_block(originalArgsBlock, &argsBlock);
		}

		// This number has been computed while registering the parent's dependencies
		size_t numDeps = parent->getMaxChildDependencies();

		Task *task = AddTask::createTask(
			parentTaskInfo, parentTaskInvocationInfo,
			argsBlock, originalArgsBlockSize,
			parent->getFlags(), numDeps, fromTaskContext
		);
		assert(task != nullptr);

		argsBlock = task->getArgsBlock();
		assert(argsBlock != nullptr);

		// Copy the args block if it was not duplicated
		if (!hasPreallocatedArgsBlock) {
			if (parentTaskInfo->duplicate_args_block != nullptr) {
				parentTaskInfo->duplicate_args_block(originalArgsBlock, &argsBlock);
			} else {
				memcpy(argsBlock, originalArgsBlock, originalArgsBlockSize);
			}
		}

		// Set bounds of grainsize
		size_t lowerBound = parentBounds.lower_bound;
		size_t upperBound = std::min(lowerBound + parentBounds.grainsize, parentBounds.upper_bound);
		parentBounds.lower_bound = upperBound;

		Taskloop *taskloop = (Taskloop *) task;
		Taskloop::bounds_t &childBounds = taskloop->getBounds();
		childBounds.lower_bound = lowerBound;
		childBounds.upper_bound = upperBound;

		// Submit task and register dependencies
		AddTask::submitTask(task, parent, fromTaskContext);
	}
};

#endif // LOOP_GENERATOR_HPP
