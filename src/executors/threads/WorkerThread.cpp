/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "CPUActivation.hpp"
#include "TaskFinalization.hpp"
#include "TaskFinalizationImplementation.hpp"
#include "ThreadManager.hpp"
#include "WorkerThread.hpp"
#include "scheduling/Scheduler.hpp"
#include "system/If0Task.hpp"
#include "system/PollingAPI.hpp"
#include "tasks/Task.hpp"

#include <DataAccessRegistration.hpp>

#include <InstrumentTaskExecution.hpp>
#include <InstrumentTaskStatus.hpp>
#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContextImplementation.hpp>
#include <InstrumentThreadManagement.hpp>

#include <atomic>

#include <pthread.h>
#include <cstring>

void WorkerThread::initialize()
{
	Instrument::createdThread(/* OUT */ _instrumentationId);
	
	assert(getComputePlace() != nullptr);
	
	Instrument::ThreadInstrumentationContext instrumentationContext((Instrument::task_id_t(), getComputePlace()->getInstrumentationId(), _instrumentationId));
	
	markAsCurrentWorkerThread();
	
	// This is needed for kernel-level threads to stop them after initialization 
	synchronizeInitialization();
	
	Instrument::threadHasResumed(_instrumentationId, getComputePlace()->getInstrumentationId());
}


void WorkerThread::body()
{
	initialize();
	
	CPU *cpu = getComputePlace();
	Instrument::ThreadInstrumentationContext instrumentationContext((Instrument::task_id_t(), cpu->getInstrumentationId(), _instrumentationId));
	
	while (!_mustShutDown) {
		CPUActivation::activationCheck(this);
		
		cpu = getComputePlace();
		instrumentationContext.updateComputePlace(cpu->getInstrumentationId());
		
		if (_task == nullptr) {
			Scheduler::polling_slot_t pollingSlot;
			
			if (Scheduler::requestPolling(cpu, &pollingSlot)) {
				Instrument::threadEnterBusyWait(Instrument::scheduling_polling_slot_busy_wait_reason);
				while ((_task == nullptr) && !ThreadManager::mustExit() && CPUActivation::acceptsWork(cpu)) {
					// Keep trying
					pollingSlot._task.compare_exchange_strong(_task, nullptr);
					if (_task == nullptr) {
						PollingAPI::handleServices();
					}
				}
				
				if (ThreadManager::mustExit()) {
					__attribute__((unused)) bool worked = Scheduler::releasePolling(cpu, &pollingSlot);
					assert(worked && "A failure to release the scheduler polling slot means that the thread has got a task assigned, however the runtime is shutting down");
				}
				Instrument::threadExitBusyWait();
				
				if (!CPUActivation::acceptsWork(cpu)) {
					// The CPU is about to be disabled
					
					// Release the polling slot
					Scheduler::releasePolling(cpu, &pollingSlot);
					
					// We may already have a task assigned through
					pollingSlot._task.compare_exchange_strong(_task, nullptr);
				}
			} else {
				// Did not receive neither the polling slot nor a task
			}
		} else {
			// The thread has been preassigned a task before being resumed
		}
		
		if (_task != nullptr) {
			WorkerThread *assignedThread = _task->getThread();
			
			// A task already assigned to another thread
			if (assignedThread != nullptr) {
				_task = nullptr;
				
				ThreadManager::addIdler(this);
				switchTo(assignedThread);
			} else {
				if (_task->isIf0()) {
					// An if0 task executed outside of the implicit taskwait of its parent (i.e. not inline)
					Task *if0Task = _task;
					
					// This is needed, since otherwise the semantics would be that the if0Task task is being launched from within its own execution
					_task = nullptr;
					
					If0Task::executeNonInline(this, if0Task, cpu);
				} else {
					handleTask(cpu);
				}
				
				_task = nullptr;
			}
		} else {
			// Try to advance work before going to sleep
			PollingAPI::handleServices();
			
			// The code below is protected by a condition because under certain CPU activation/deactivation
			// cases, the call to CPUActivation::activationCheck may have put the thread in the idle queue
			// and the shutdown mechanism may have waken up the thread. In that case we do not want the
			// thread to go back to the idle queue. The previous case does not need the condition because
			// there is a task to be run and thus the program cannot be performing (a regular) shutdown.
			if (!_mustShutDown) {
				ThreadManager::addIdler(this);
				switchTo(nullptr);
			}
		}
	}
	
	Instrument::threadWillShutdown();
	
	shutdownSequence();
}


void WorkerThread::handleTask(CPU *cpu)
{
	_task->setThread(this);
	
	Instrument::task_id_t taskId = _task->getInstrumentationTaskId();
	
	Instrument::ThreadInstrumentationContext instrumentationContext(taskId, cpu->getInstrumentationId(), _instrumentationId);
	
	if (_task->hasCode()) {
		Instrument::startTask(taskId);
		Instrument::taskIsExecuting(taskId);
		
		// Run the task
		std::atomic_thread_fence(std::memory_order_acquire);
		_task->body();
		std::atomic_thread_fence(std::memory_order_release);
		
		Instrument::taskIsZombie(taskId);
		Instrument::endTask(taskId);
	}
	
	// Update the CPU since the thread may have migrated
	cpu = getComputePlace();
	
	// The release must be delayed until all children has finished
	if (_task->mustDelayDataAccessRelease()) {
		_task->setDelayedDataAccessRelease(true);
		DataAccessRegistration::handleEnterTaskwait(_task, cpu);
		if (!_task->markAsFinished()) {
			_task = nullptr;
			return;
		}
		
		DataAccessRegistration::handleExitTaskwait(_task, cpu);
		_task->increaseRemovalBlockingCount();
	}
	
	// Release the accesses
	DataAccessRegistration::unregisterTaskDataAccesses(_task, cpu);
	
	// Try to dispose the task
	if (_task->markAsFinishedAfterDataAccessRelease()) {
		TaskFinalization::disposeOrUnblockTask(_task, cpu);
	}
	
	_task = nullptr;
}


