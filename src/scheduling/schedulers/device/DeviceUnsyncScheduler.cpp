/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2021 Barcelona Supercomputing Center (BSC)
*/

#include "DeviceUnsyncScheduler.hpp"

Task *DeviceUnsyncScheduler::getReadyTask(ComputePlace *computePlace, bool &hasIncompatibleWork)
{
	hasIncompatibleWork = false;

	// Check if there is work remaining in the ready queue.
	Task *task = regularGetReadyTask(computePlace);
	assert(task == nullptr || !task->isTaskfor());

	return task;
}
