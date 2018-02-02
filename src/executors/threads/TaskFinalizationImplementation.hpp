/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_FINALIZATION_IMPLEMENTATION_HPP
#define TASK_FINALIZATION_IMPLEMENTATION_HPP

#include "DataAccessRegistration.hpp"
#include "TaskFinalization.hpp"

#include <InstrumentTaskStatus.hpp>


void TaskFinalization::disposeOrUnblockTask(Task *task, ComputePlace *computePlace)
{
	bool readyOrDisposable = true;
	
	// Follow up the chain of ancestors and dispose them as needed and wake up any in a taskwait that finishes in this moment
	while ((task != nullptr) && readyOrDisposable) {
		Task *parent = task->getParent();
		
		if (task->hasDelayedDataAccessRelease()) {
			DataAccessRegistration::handleExitTaskwait(task, computePlace);
			
			// Unregister data accesses preventing the removal of the task
			task->increaseRemovalBlockingCount();
			DataAccessRegistration::unregisterTaskDataAccesses(task, computePlace);
			if (!task->markAsFinishedAfterDataAccessRelease()) {
				break;
			}
		}
		assert(!task->hasDelayedDataAccessRelease());
		
		if (task->hasFinished()) {
			// NOTE: Handle task removal before unlinking from parent
			DataAccessRegistration::handleTaskRemoval(task, computePlace);
			
			readyOrDisposable = task->unlinkFromParent();
			Instrument::destroyTask(task->getInstrumentationTaskId());
			
			// NOTE: The memory layout is defined in nanos_create_task
			void *disposableBlock = nullptr;
			if (task->isArgsBlockOwner()) {
				disposableBlock = task->getArgsBlock();
			} else {
				assert(task->isTaskloop());
				disposableBlock = (void *)task;
			}
			assert(disposableBlock != nullptr);
			
			Instrument::taskIsBeingDeleted(task->getInstrumentationTaskId());
			task->~Task();
			free(disposableBlock); // FIXME: Need a proper object recycling mechanism here
			task = parent;
			
			// A task without parent must be a spawned function
			if (parent == nullptr) {
				SpawnedFunctions::_pendingSpawnedFunctions--;
			}
		} else {
			assert(!task->hasFinished());
			
			// An ancestor in a taskwait that finishes at this point
			Scheduler::taskGetsUnblocked(task, computePlace);
			readyOrDisposable = false;
		}
	}
}


#endif // TASK_FINALIZATION_IMPLEMENTATION_HPP
