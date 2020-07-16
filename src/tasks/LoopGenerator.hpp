/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
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
	static inline Taskfor *createCollaborator(Taskfor *parent, ComputePlace *computePlace)
	{
		assert(parent != nullptr);

		nanos6_task_info_t *parentTaskInfo = parent->getTaskInfo();
		nanos6_task_invocation_info_t *parentTaskInvocationInfo = parent->getTaskInvokationInfo();
		Instrument::task_id_t parentTaskInstrumentationId = parent->getInstrumentationTaskId();
		size_t flags = parent->getFlags();
		void *originalArgsBlock = parent->getArgsBlock();
		size_t originalArgsBlockSize = parent->getArgsBlockSize();

		Instrument::task_id_t taskId =
			Instrument::enterInitTaskforCollaborator(
				parentTaskInstrumentationId,
				parentTaskInfo,
				parentTaskInvocationInfo,
				flags
			);

		Taskfor *taskfor = computePlace->getPreallocatedTaskfor();
		assert(taskfor != nullptr);

		void *argsBlock = nullptr;
		bool hasPreallocatedArgsBlock = parent->hasPreallocatedArgsBlock();
		if (hasPreallocatedArgsBlock) {
			assert(parentTaskInfo->duplicate_args_block != nullptr);
			parentTaskInfo->duplicate_args_block(originalArgsBlock, &argsBlock);
		} else {
			argsBlock = computePlace->getPreallocatedArgsBlock(originalArgsBlockSize);
		}
		assert(argsBlock != nullptr);

		taskfor->reinitialize(
			argsBlock, originalArgsBlockSize,
			parentTaskInfo,
			parentTaskInvocationInfo,
			nullptr, taskId, flags
		);

		// Copy the args block if it was not duplicated
		if (!hasPreallocatedArgsBlock) {
			if (parentTaskInfo->duplicate_args_block != nullptr) {
				parentTaskInfo->duplicate_args_block(originalArgsBlock, &argsBlock);
			} else {
				memcpy(argsBlock, originalArgsBlock, originalArgsBlockSize);
			}
		}

		// In case this has been created by a taskloop for, and we received
		// the taskloop flag, remove it. Otherwise, we may end up disposing
		// a preallocated taskfor
		taskfor->setTaskloop(false);

		// Set the parent
		taskfor->setParent(parent);

		// Instrument the task creation
		Instrument::task_id_t taskInstrumentationId = taskfor->getInstrumentationTaskId();
		Instrument::exitInitTaskforCollaborator(parentTaskInstrumentationId, taskInstrumentationId);

		return taskfor;
	}

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

		// Avoid creating a taskloop when dealing with taskloop fors
		size_t flags = parent->getFlags();
		if (parent->isTaskfor()) {
			flags &= ~nanos6_taskloop_task;
		}

		void *argsBlock = nullptr;
		bool hasPreallocatedArgsBlock = parent->hasPreallocatedArgsBlock();
		if (hasPreallocatedArgsBlock) {
			assert(parentTaskInfo->duplicate_args_block != nullptr);
			parentTaskInfo->duplicate_args_block(originalArgsBlock, &argsBlock);
		}

		// Create the task with undefined number of dependencies
		Task *task = AddTask::createTask(
			parentTaskInfo, parentTaskInvocationInfo,
			argsBlock, originalArgsBlockSize,
			flags, -1, fromTaskContext
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

		// Both taskfor and taskloop bounds share the same data structure
		if (parent->isTaskfor()) {
			Taskfor *taskfor = (Taskfor *) task;
			taskfor->initialize(lowerBound, upperBound, parentBounds.chunksize);
		} else {
			Taskloop *taskloop = (Taskloop *) task;
			Taskloop::bounds_t &childBounds = taskloop->getBounds();
			childBounds.lower_bound = lowerBound;
			childBounds.upper_bound = upperBound;
		}

		// Submit task and register dependencies
		AddTask::submitTask(task, parent, fromTaskContext);
	}
};

#endif // LOOP_GENERATOR_HPP
