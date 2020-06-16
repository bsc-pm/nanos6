/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_FINALIZATION_IMPLEMENTATION_HPP
#define TASK_FINALIZATION_IMPLEMENTATION_HPP

#include "DataAccessRegistration.hpp"
#include "MemoryAllocator.hpp"
#include "TaskDataAccesses.hpp"
#include "TaskFinalization.hpp"
#include "hardware-counters/TaskHardwareCounters.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/StreamManager.hpp"
#include "tasks/Taskfor.hpp"
#include "tasks/Taskloop.hpp"

#include <InstrumentComputePlaceId.hpp>
#include <InstrumentTaskExecution.hpp>
#include <InstrumentTaskStatus.hpp>
#include <InstrumentThreadId.hpp>
#include <Monitoring.hpp>

void TaskFinalization::taskFinished(Task *task, ComputePlace *computePlace, bool fromBusyThread)
{
	assert(task != nullptr);
	//! Decrease the _countdownToBeWokenUp of the task, which was initialized to 1.
	//! If it then becomes 0, we can propagate the counter through its parents.
	bool ready = task->finishChild();

	//! We always use a local CPUDependencyData struct here to avoid issues
	//! with re-using an already used CPUDependencyData
	CPUDependencyData *localHpDependencyData = nullptr;

	while ((task != nullptr) && ready) {
		Task *parent = task->getParent();

		// If this is the first iteration of the loop, the task will test true to hasFinished and false to mustDelayRelease, doing
		// nothing inside the conditionals.
		if (task->hasFinished()) {
			// Complete the delayed release of dependencies of the task if it has a wait clause
			if (task->mustDelayRelease()) {
				if (task->markAllChildrenAsFinished(computePlace)) {
					if (!localHpDependencyData) {
						localHpDependencyData = new CPUDependencyData();
					}

					DataAccessRegistration::unregisterTaskDataAccesses(
						task, computePlace,
						*localHpDependencyData,
						/* memory place */ nullptr,
						fromBusyThread);

					task->setComputePlace(nullptr);

					Monitoring::taskFinished(task);
					// This is just to emulate a recursive call to TaskFinalization::taskFinished() again.
					// It should not return false because at this point delayed release has happenned which means that
					// the task has gone through a taskwait (no more children should be unfinished)
					ready = task->finishChild();
					assert(ready);
					if (task->markAsReleased())
						TaskFinalization::disposeTask(task);
				}

				assert(!task->mustDelayRelease());
			} else if (task->isTaskforCollaborator()) {
				Taskfor *collaborator = (Taskfor *)task;
				Taskfor *source = (Taskfor *)parent;

				size_t completedIts = collaborator->getCompletedIterations();
				if (completedIts > 0) {
					bool finishedSource = source->decrementRemainingIterations(completedIts);
					if (finishedSource) {
						source->markAsFinished(computePlace);
						assert(computePlace != nullptr);

						DataAccessRegistration::unregisterTaskDataAccesses(
							source, computePlace,
							computePlace->getDependencyData());

						// There is one count for the finished source, but we need ready = true to decrement it later again.
						ready = source->finishChild();
						assert(!ready);
						ready = true;
						if (source->markAsReleased())
							TaskFinalization::disposeTask(source);
					}
				}
			}
		} else {
			// An ancestor in a taskwait that finishes at this point
			Scheduler::addReadyTask(task, computePlace, UNBLOCKED_TASK_HINT);

			ready = false;
		}

		// Using 'task' here is forbidden, as it may have been disposed.
		if (ready && parent != nullptr) {
			ready = parent->finishChild();
		}

		task = parent;
	}

	if (localHpDependencyData) {
		delete localHpDependencyData;
	}
}

void TaskFinalization::disposeTask(Task *task)
{
	bool disposable = true;

	// Follow up the chain of ancestors and dispose them as needed and wake up any in a taskwait that finishes in this moment
	while ((task != nullptr) && disposable) {
		Task *parent = task->getParent();

		assert(task->hasFinished());

		disposable = task->unlinkFromParent();
		bool isTaskfor = task->isTaskfor();
		bool isTaskloop = task->isTaskloop();
		bool isSpawned = task->isSpawned();
		bool isStreamExecutor = task->isStreamExecutor();

		if (task->isDisposable()) {
			Instrument::destroyTask(task->getInstrumentationTaskId());
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

			// taskloop and taskfor flags can both be enabled for the same task.
			// If this is the case, it means we are dealing with taskloop for,
			// which is a taskloop that generates taskfors. Thus, we must treat
			// the task as a taskloop. It is important to check taskloop condition
			// before taskfor one, to dispose a taskloop in the case of taskloop for.
			if (isTaskloop) {
				disposableBlockSize += sizeof(Taskloop);
			} else if (isTaskfor) {
				disposableBlockSize += sizeof(Taskfor);
			} else if (isStreamExecutor) {
				disposableBlockSize += sizeof(StreamExecutor);
			} else {
				disposableBlockSize += sizeof(Task);
			}

			TaskDataAccesses &dataAccesses = task->getDataAccesses();
			disposableBlockSize += dataAccesses.getAdditionalMemorySize();
			disposableBlockSize += TaskHardwareCounters::getTaskHardwareCountersSize();

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

			// taskloop and taskfor flags can both be enabled for the same task.
			// If this is the case, it means we are dealing with taskloop for,
			// which is a taskloop that generates taskfors. Thus, we must treat
			// the task as a taskloop. It is important to check taskloop condition
			// before taskfor one, to dispose a taskloop in the case of taskloop for.
			if (isTaskloop) {
				((Taskloop *)task)->~Taskloop();
			} else if (isTaskfor) {
				((Taskfor *)task)->~Taskfor();
			} else if (isStreamExecutor) {
				((StreamExecutor *)task)->~StreamExecutor();
			} else {
				task->~Task();
			}
			MemoryAllocator::free(disposableBlock, disposableBlockSize);
		} else {
			// Although collaborators cannot be disposed, they must destroy their
			// args blocks. The destroy function free the memory of the args block
			// in case the collaborator has preallocated args block; otherwise the
			// args block is just destroyed calling the destructors
			nanos6_task_info_t *taskInfo = task->getTaskInfo();
			if (taskInfo->destroy_args_block != nullptr) {
				taskInfo->destroy_args_block(task->getArgsBlock());
			}
		}

		task = parent;

		if (isSpawned) {
			SpawnedFunctions::_pendingSpawnedFunctions--;
		} else if (isStreamExecutor) {
			StreamManager::_activeStreamExecutors--;
		}
	}
}


#endif // TASK_FINALIZATION_IMPLEMENTATION_HPP
