/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "Taskloop.hpp"
#include "tasks/LoopGenerator.hpp"

void Taskloop::body(
	__attribute__((unused)) void *deviceEnvironment,
	__attribute__((unused)) nanos6_address_translation_entry_t *translationTable
) {

	nanos6_task_info_t *taskInfo = getTaskInfo();
	bool isChildTaskloop = isSourceTaskloop();

	if (isChildTaskloop) {
		taskInfo->implementations[0].run(getArgsBlock(), &getBounds(), nullptr);
	}
	else {
		nanos6_task_invocation_info_t *taskInvocationInfo = getTaskInvokationInfo();
		void *originalArgsBlock = getArgsBlock();
		size_t originalArgsBlockSize = getArgsBlockSize();
		size_t flags = getFlags();
		bool preallocatedArgsBlock = hasPreallocatedArgsBlock();

		while (hasPendingIterations()) {
			LoopGenerator::createTaskloopExecutor(taskInfo, taskInvocationInfo, originalArgsBlockSize, originalArgsBlock, flags, preallocatedArgsBlock, _bounds);
		}
	}
}
