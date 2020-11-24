/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include "DeviceUnsyncScheduler.hpp"

Task *DeviceUnsyncScheduler::getReadyTask(ComputePlace *computePlace)
{
	Task *task = nullptr;

	// 1. Check if there is an immediate successor.
	if (_enableImmediateSuccessor && computePlace != nullptr) {
		size_t immediateSuccessorId = computePlace->getIndex();
		if (_immediateSuccessorTasks[immediateSuccessorId] != nullptr) {
			task = _immediateSuccessorTasks[immediateSuccessorId];
			assert(!task->isTaskfor());
			_immediateSuccessorTasks[immediateSuccessorId] = nullptr;
			return task;
		}
	}

	// 2. Check if there is work remaining in the ready queue.
	task = regularGetReadyTask(computePlace);

	// 3. Try to get work from other immediateSuccessorTasks.
	if (task == nullptr && _enableImmediateSuccessor) {
		for (size_t i = 0; i < _immediateSuccessorTasks.size(); i++) {
			if (_immediateSuccessorTasks[i] != nullptr) {
				task = _immediateSuccessorTasks[i];
				assert(!task->isTaskfor());
				_immediateSuccessorTasks[i] = nullptr;
				break;
			}
		}
	}

	assert(task == nullptr || !task->isTaskfor());

	return task;
}
