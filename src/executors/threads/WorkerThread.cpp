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
#include "tasks/TaskImplementation.hpp"

#include <DataAccessRegistration.hpp>

#include <InstrumentComputePlaceManagement.hpp>
#include <InstrumentTaskExecution.hpp>
#include <InstrumentTaskStatus.hpp>
#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContextImplementation.hpp>
#include <InstrumentThreadManagement.hpp>
#include <instrument/support/InstrumentThreadLocalDataSupport.hpp>
#include <instrument/support/InstrumentThreadLocalDataSupportImplementation.hpp>

#include <atomic>

#include <alloca.h>
#include <pthread.h>
#include <cstring>

void WorkerThread::initialize()
{
	Instrument::createdThread(_instrumentationId, getComputePlace()->getInstrumentationId());
	
	assert(getComputePlace() != nullptr);
	
	Instrument::ThreadInstrumentationContext instrumentationContext(Instrument::task_id_t(), getComputePlace()->getInstrumentationId(), _instrumentationId);
	
	markAsCurrentWorkerThread();
	
	// This is needed for kernel-level threads to stop them after initialization 
	synchronizeInitialization();
	
	Instrument::threadHasResumed(_instrumentationId, getComputePlace()->getInstrumentationId());
}


void WorkerThread::body()
{
	initialize();
	
	CPU *cpu = getComputePlace();
	Instrument::ThreadInstrumentationContext instrumentationContext(Instrument::task_id_t(), cpu->getInstrumentationId(), _instrumentationId);
	
	while (!_mustShutDown) {
		CPUActivation::activationCheck(this);
		
		cpu = getComputePlace();
		instrumentationContext.updateComputePlace(cpu->getInstrumentationId());
		
		if (_task == nullptr) {
			_task = Scheduler::getReadyTask(cpu, nullptr, true);
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
				
				Instrument::suspendingComputePlace(cpu->getInstrumentationId());
				switchTo(nullptr);
				cpu = getComputePlace();
				Instrument::resumedComputePlace(cpu->getInstrumentationId());
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
		nanos6_address_translation_entry_t *translationTable = nullptr;
		
		nanos6_task_info_t const * const taskInfo = _task->getTaskInfo();
		if (taskInfo->num_symbols >= 0) {
			translationTable = (nanos6_address_translation_entry_t *) alloca(sizeof(nanos6_address_translation_entry_t) * taskInfo->num_symbols);
			
			for (int index = 0; index < taskInfo->num_symbols; index++) {
				translationTable[index] = {0, 0};
			}
		}
		
		Instrument::startTask(taskId);
		Instrument::taskIsExecuting(taskId);
		
		// Run the task
		std::atomic_thread_fence(std::memory_order_acquire);
		_task->body(nullptr, translationTable);
		std::atomic_thread_fence(std::memory_order_release);
		
		Instrument::taskIsZombie(taskId);
		Instrument::endTask(taskId);
	}
	
	// Update the CPU since the thread may have migrated
	cpu = getComputePlace();
	instrumentationContext.updateComputePlace(cpu->getInstrumentationId());
	
	if (_task->markAsFinished(cpu)) {
		DataAccessRegistration::unregisterTaskDataAccesses(
			_task,
			cpu,
			cpu->getDependencyData()
		);
		
		if (_task->markAsReleased()) {
			TaskFinalization::disposeOrUnblockTask(_task, cpu);
		}
	}
	
	_task = nullptr;
}


