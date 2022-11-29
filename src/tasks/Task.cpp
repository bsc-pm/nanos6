/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2021-2022 Barcelona Supercomputing Center (BSC)
*/

#include "nanos6.h"

#include "Task.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


void Task::runOnready(WorkerThread *currentThread)
{
	assert(currentThread != nullptr);
	assert(_taskInfo != nullptr);
	assert(_taskInfo->onready_action != nullptr);

	Task *currentTask = currentThread->unassignTask();
	currentThread->setTask(this);

	// Execute the onready action function
	_taskInfo->onready_action(_argsBlock);

	currentThread->unassignTask();
	currentThread->setTask(currentTask);
}
