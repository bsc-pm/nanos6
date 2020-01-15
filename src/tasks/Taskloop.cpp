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
	bool isChildTaskloop = !isSourceTaskloop();

	if (isChildTaskloop) {
		taskInfo->implementations[0].run(getArgsBlock(), &getBounds(), nullptr);
	}
	else {
		while (hasPendingIterations()) {
			LoopGenerator::createTaskloopExecutor(this, _bounds);
		}
	}
}
