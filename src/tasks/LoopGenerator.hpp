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

		assert(parentTaskInfo->implementation_count == 1); //TODO: Temporary check until multiple implementations are supported
		assert(argsBlock != nullptr);
		assert(taskfor_ptr != nullptr);

		Instrument::task_id_t taskId = Instrument::enterAddTaskforCollaborator(parentTaskInstrumentationId, parentTaskInfo, parentTaskInvocationInfo, flags);

		bool isTaskfor = flags & nanos6_task_flag_t::nanos6_taskfor_task;
		FatalErrorHandler::failIf(!isTaskfor, "Only taskfors can be created this way.");

		taskfor->reinitialize(argsBlock, originalArgsBlockSize, parentTaskInfo, parentTaskInvocationInfo, nullptr, taskId, flags);

		assert(argsBlock != nullptr);
		assert(taskfor_ptr != nullptr);

		// Copy the args block if it was not duplicated
		if (!hasPreallocatedArgsBlock) {
			if (parentTaskInfo->duplicate_args_block != nullptr) {
				parentTaskInfo->duplicate_args_block(originalArgsBlock, &argsBlock);
			} else {
				memcpy(argsBlock, originalArgsBlock, originalArgsBlockSize);
			}
		}

		// Set the flags
		taskfor->setRunnable(true);

		// In case this has been created by a taskloop for, and we received the taskloop flag, remove it.
		// Otherwise, we may end up disposing a preallocated taskfor.
		taskfor->setTaskloop(false);

		// Set the parent
		taskfor->setParent(parent);

		// Instrument the task creation
		Instrument::task_id_t taskInstrumentationId = taskfor->getInstrumentationTaskId();
		Instrument::exitAddTaskforCollaborator(parentTaskInstrumentationId, taskInstrumentationId);

		return taskfor;
	}

	static inline void createTaskloopExecutor(Taskloop *parent, Taskloop::bounds_t &parentBounds)
	{
		assert(parent != nullptr);

		nanos6_task_info_t *parentTaskInfo = parent->getTaskInfo();
		nanos6_task_invocation_info_t *parentTaskInvocationInfo = parent->getTaskInvokationInfo();
		void *originalArgsBlock = parent->getArgsBlock();
		size_t originalArgsBlockSize = parent->getArgsBlockSize();
		size_t flags = parent->getFlags();

		void *argsBlock = nullptr;
		Task *task = nullptr;
		bool hasPreallocatedArgsBlock = parent->hasPreallocatedArgsBlock();
		if (hasPreallocatedArgsBlock) {
			assert(parentTaskInfo->duplicate_args_block != nullptr);
			parentTaskInfo->duplicate_args_block(originalArgsBlock, &argsBlock);
		}

		// We are dealing with a taskloop for. That means instead of regular tasks, we must create taskfors.
		// For that purpose, we must remove the taskloop flag, otherwise regular tasks will be created.
		if (parent->isTaskfor()) {
			flags &= ~nanos6_task_flag_t::nanos6_taskloop_task;
		}

		nanos6_create_task(parentTaskInfo, parentTaskInvocationInfo, originalArgsBlockSize, &argsBlock, (void **)&task, flags, -1);
		assert(argsBlock != nullptr);
		assert(task != nullptr);

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
		parentBounds.lower_bound = std::min(parentBounds.lower_bound + parentBounds.grainsize, parentBounds.upper_bound);
		size_t upperBound = parentBounds.lower_bound;
		// Both taskfor and taskloop bounds share the same data structure.
		if (parent->isTaskfor()) {
			Taskfor *taskfor = (Taskfor *) task;
			taskfor->initialize(lowerBound, upperBound, parentBounds.chunksize);
		} else {
			Taskloop *taskloop = (Taskloop *) task;
			Taskloop::bounds_t &childBounds = taskloop->getBounds();
			childBounds.lower_bound = lowerBound;
			childBounds.upper_bound = upperBound;
		}

		// Register deps
		nanos6_submit_task((void *)task);
	}
};

#endif // LOOP_GENERATOR_HPP
