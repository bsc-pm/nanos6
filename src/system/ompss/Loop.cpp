/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6.h>

#include "executors/threads/WorkerThread.hpp"
#include "tasks/Taskfor.hpp"
#include "tasks/Taskloop.hpp"

#include <cassert>

void nanos6_register_loop_bounds(
	void *taskHandle,
	size_t lower_bound,
	size_t upper_bound,
	size_t grainsize,
	size_t chunksize
) {
	Task *task = (Task *) taskHandle;
	assert(task != nullptr);
	assert(task->isTaskfor() || task->isTaskloop());

	if (task->isTaskloop()) {
		Taskloop *taskloop = (Taskloop *) task;
		taskloop->initialize(lower_bound, upper_bound, grainsize);
	}
	else {
		Taskfor *taskfor = (Taskfor *) task;
		taskfor->initialize(lower_bound, upper_bound, chunksize);
	}
}

