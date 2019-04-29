/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
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
			_immediateSuccessorTasks[immediateSuccessorId] = nullptr;
			assert(!task->isTaskloop());
			FatalErrorHandler::failIf(task->isTaskloop(), "Devices do not support task fors.");
			return task;
		}
	}
	
	// 2. Check if there is work remaining in the ready queue.
	task = _readyTasks->getReadyTask(computePlace);
	
	 //3. Try to get work from other immediateSuccessorTasks.
	if (task == nullptr && _enableImmediateSuccessor) {
		for (size_t i = 0; i < _immediateSuccessorTasks.size(); i++) {
			if (_immediateSuccessorTasks[i] != nullptr) {
				task = _immediateSuccessorTasks[i];
				assert(!task->isTaskloop());
				FatalErrorHandler::failIf(task->isTaskloop(), "Devices do not support task fors.");
				_immediateSuccessorTasks[i] = nullptr;
				break;
			}
		}
	}
	
	assert(task == nullptr || !task->isTaskloop());
	FatalErrorHandler::failIf(task != nullptr && task->isTaskloop(), "Devices do not support task fors.");
	
	return task;
}
