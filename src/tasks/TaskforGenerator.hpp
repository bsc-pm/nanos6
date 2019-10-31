/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASKFOR_GENERATOR_HPP
#define TASKFOR_GENERATOR_HPP

#include "Taskfor.hpp"
#include "system/ompss/AddTask.hpp"

#include <InstrumentAddTask.hpp>
#include <InstrumentTaskId.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>


class TaskforGenerator {
public:
	static inline Taskfor *createCollaborator(Taskfor *parent, TaskforInfo::bounds_t &bounds, ComputePlace *computePlace)
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
		taskfor->getTaskforInfo().setBounds(bounds);

		// Instrument the task creation
		Instrument::task_id_t taskInstrumentationId = taskfor->getInstrumentationTaskId();
		Instrument::exitAddTaskforCollaborator(parentTaskInstrumentationId, taskInstrumentationId);

		return taskfor;
	}
};

#endif // TASKFOR_GENERATOR_HPP
