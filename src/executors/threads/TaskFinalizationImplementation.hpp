/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_FINALIZATION_IMPLEMENTATION_HPP
#define TASK_FINALIZATION_IMPLEMENTATION_HPP

#include "DataAccessRegistration.hpp"
#include "MemoryAllocator.hpp"
#include "TaskFinalization.hpp"
#include "executors/threads/CPUManager.hpp"
#include "tasks/StreamManager.hpp"
#include "tasks/Taskfor.hpp"
#include "TaskDataAccesses.hpp"

#include <HardwareCounters.hpp>
#include <InstrumentComputePlaceId.hpp>
#include <InstrumentTaskExecution.hpp>
#include <InstrumentTaskStatus.hpp>
#include <InstrumentThreadId.hpp>
#include <Monitoring.hpp>

void TaskFinalization::disposeOrUnblockTask(Task *task, ComputePlace *computePlace, bool fromBusyThread)
{
	bool readyOrDisposable = true;
	
	//! We always use a local CPUDependencyData struct here to avoid issues
	//! with re-using an already used CPUDependencyData
	CPUDependencyData localHpDependencyData;
	
	// Follow up the chain of ancestors and dispose them as needed and wake up any in a taskwait that finishes in this moment
	while ((task != nullptr) && readyOrDisposable) {
		Task *parent = task->getParent();
		
		// Complete the delayed release of dependencies of the task if it has a wait clause
		if (task->hasFinished() && task->mustDelayRelease()) {
			readyOrDisposable = false;
			if (task->markAllChildrenAsFinished(computePlace)) {
				DataAccessRegistration::unregisterTaskDataAccesses(
					task, computePlace,
					localHpDependencyData,
					/* memory place */ nullptr,
					fromBusyThread
				);
				
				Monitoring::taskFinished(task);
				HardwareCounters::taskFinished(task);
				
				task->setComputePlace(nullptr);
				
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
			if (task->isTaskfor() && task->isRunnable()) {
				assert(!readyOrDisposable);
				Taskfor *collaborator = (Taskfor *) task;
				Taskfor *source = (Taskfor *) parent;
				
				if (source->decrementRemainingIterations(collaborator->getCompletedIterations())) {
					assert(!source->hasPendingIterations());
					
					__attribute__((unused)) bool finished = source->markAsFinished(computePlace);
					assert(finished);
					DataAccessRegistration::unregisterTaskDataAccesses(source, computePlace, localHpDependencyData);
					readyOrDisposable = source->markAsReleased();
				}
			}
			
			bool isTaskfor = task->isTaskfor();
			bool isSpawned = task->isSpawned();
			bool isStreamExecutor = task->isStreamExecutor();
			
			Instrument::destroyTask(task->getInstrumentationTaskId());
			
			//! We cannot destroy collaborator tasks because they are preallocated tasks that are used during all the program execution.
			//! However, we must destroy taskfors that are not collaborators. A collaborator must be runnable. Thus, if a taskfor is runnable,
			//! it is a collaborator, so no destroy. Otherwise, it is not a collaborator, so must be destroyed.
			bool destroy = !(task->isTaskfor() && task->isRunnable());
			
			if (destroy) {
				// NOTE: The memory layout is defined in nanos6_create_task
				void *disposableBlock;
				size_t disposableBlockSize;
				
				if (task->hasPreallocatedArgsBlock()) {
					disposableBlock = task;
					disposableBlockSize = 0;
				} else {
					disposableBlock = task->getArgsBlock();
					assert(disposableBlock != nullptr);
					
					disposableBlockSize = (char *)task - (char *)disposableBlock;
				}
				
				if (isTaskfor) {
					disposableBlockSize += sizeof(Taskfor);
				} else if (isStreamExecutor) {
					disposableBlockSize += sizeof(StreamExecutor);
				} else {
					disposableBlockSize += sizeof(Task);
				}
				
				TaskDataAccesses &dataAccesses = task->getDataAccesses();
				disposableBlockSize += dataAccesses.getAdditionalMemorySize();
				
				Instrument::taskIsBeingDeleted(task->getInstrumentationTaskId());
				
				// Call the taskinfo destructor if not null
				nanos6_task_info_t *taskInfo = task->getTaskInfo();
				if (taskInfo->destroy_args_block != nullptr) {
					taskInfo->destroy_args_block(task->getArgsBlock());
				}
				
				StreamFunctionCallback *spawnCallback = task->getParentSpawnCallback();
				if (spawnCallback != nullptr) {
					StreamExecutor *executor = (StreamExecutor *)(task->getParent());
					assert(executor != nullptr);
					executor->decreaseCallbackParticipants(spawnCallback);
				}
				
				if (isTaskfor) {
					((Taskfor *)task)->~Taskfor();
				} else if (isStreamExecutor) {
					((StreamExecutor *)task)->~StreamExecutor();
				} else {
					task->~Task();
				}
				MemoryAllocator::free(disposableBlock, disposableBlockSize);
			} else {
				Instrument::taskIsBeingDeleted(task->getInstrumentationTaskId());
			}
			
			task = parent;
			
			if (isSpawned) {
				SpawnedFunctions::_pendingSpawnedFunctions--;
			} else if (isStreamExecutor) {
				StreamManager::_activeStreamExecutors--;
			}
		} else {
			assert(!task->hasFinished());
			
			// An ancestor in a taskwait that finishes at this point
			Scheduler::addReadyTask(task, computePlace, UNBLOCKED_TASK_HINT);
			
			// After adding a task, the CPUManager may want to unidle CPUs
			CPUManager::executeCPUManagerPolicy(computePlace, ADDED_TASKS, 1);
			
			readyOrDisposable = false;
		}
	}
}


#endif // TASK_FINALIZATION_IMPLEMENTATION_HPP
