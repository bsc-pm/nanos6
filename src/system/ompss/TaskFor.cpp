/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include <nanos6.h>

#include "executors/threads/WorkerThread.hpp"
#include "tasks/Task.hpp"
#include "tasks/Taskfor.hpp"
#include "tasks/TaskImplementation.hpp"

void nanos6_register_taskloop_bounds(
	void *taskHandle,
	size_t lower_bound,
	size_t upper_bound,
	size_t step,
	size_t chunksize
) {
	Task *task = (Task *) taskHandle;
	assert(task != nullptr);
	assert(task->isTaskfor());
	
	Taskfor *taskfor = (Taskfor *) task;
	taskfor->getTaskforInfo().initialize(lower_bound, upper_bound, step, chunksize);
}

