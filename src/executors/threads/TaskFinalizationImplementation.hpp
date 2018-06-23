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
		
		// Complete the delayed release of dependencies of the task if it has a wait clause
		if (task->hasFinished() && task->mustDelayRelease()) {
			readyOrDisposable = false;
			if (task->markAllChildrenAsFinished(computePlace)) {
				DataAccessRegistration::unregisterTaskDataAccesses(task, computePlace);
				if (task->markAsReleased()) {
					readyOrDisposable = true;
				}
			}
			assert(!task->mustDelayRelease());
			if (!readyOrDisposable)
				break;
		}
		
		if (task->hasFinished()) {
			// NOTE: Handle task removal before unlinking from parent
			DataAccessRegistration::handleTaskRemoval(task, computePlace);
			
			readyOrDisposable = task->unlinkFromParent();
			Instrument::destroyTask(task->getInstrumentationTaskId());
			
			// NOTE: The memory layout is defined in nanos_create_task
			void *disposableBlock = task->getArgsBlock();
			assert(disposableBlock != nullptr);
			
			Instrument::taskIsBeingDeleted(task->getInstrumentationTaskId());
			
			// Call the taskinfo destructor if not null
			nanos_task_info * taskInfo = task->getTaskInfo();
			if (taskInfo->destroy != nullptr) {
				taskInfo->destroy(task->getArgsBlock());
			}
			
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
			
			if (computePlace->getType() != nanos6_device_t::nanos6_host_device) {
				CPU *idleCPU = (CPU *) Scheduler::getIdleComputePlace();
				if (idleCPU != nullptr) {
					ThreadManager::resumeIdle(idleCPU);
				}
			}
		}
	}
}


#endif // TASK_FINALIZATION_IMPLEMENTATION_HPP
