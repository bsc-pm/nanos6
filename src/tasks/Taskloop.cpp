/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include "Taskloop.hpp"
#include "tasks/LoopGenerator.hpp"

void Taskloop::body(nanos6_address_translation_entry_t *translationTable)
{
	if (!isTaskloopSource()) {
		getTaskInfo()->implementations[0].run(getArgsBlock(), &getBounds(), translationTable);
	} else {
		while (getIterationCount() > 0) {
			LoopGenerator::createTaskloopExecutor(this, _bounds);
		}
	}
}
