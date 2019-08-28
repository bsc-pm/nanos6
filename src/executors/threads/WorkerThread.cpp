/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <atomic>
#include <alloca.h>
#include <atomic>
#include <cstring>
#include <pthread.h>

#include "CPUActivation.hpp"
#include "TaskFinalization.hpp"
#include "TaskFinalizationImplementation.hpp"
#include "ThreadManager.hpp"
#include "WorkerThread.hpp"
#include "hardware/HardwareInfo.hpp"
#include "scheduling/Scheduler.hpp"
#include "system/If0Task.hpp"
#include "system/PollingAPI.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"

#include <DataAccessRegistration.hpp>
#include <ExecutionWorkflow.hpp>
#include <HardwareCounters.hpp>
#include <InstrumentComputePlaceManagement.hpp>
#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentTaskExecution.hpp>
#include <InstrumentTaskStatus.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <InstrumentThreadManagement.hpp>
#include <Monitoring.hpp>

void WorkerThread::initialize()
{
	Instrument::createdThread(_instrumentationId, getComputePlace()->getInstrumentationId());
	
	assert(getComputePlace() != nullptr);
	
	Instrument::ThreadInstrumentationContext instrumentationContext(Instrument::task_id_t(), getComputePlace()->getInstrumentationId(), _instrumentationId);
	
	markAsCurrentWorkerThread();
	
	// This is needed for kernel-level threads to stop them after initialization
	synchronizeInitialization();
	
	Instrument::threadHasResumed(_instrumentationId, getComputePlace()->getInstrumentationId());
	
	HardwareCounters::initializeThread();
	Monitoring::initializeThread();
}


void WorkerThread::body()
{
	initialize();
	
	CPU *cpu = getComputePlace();
	
	Instrument::ThreadInstrumentationContext instrumentationContext(Instrument::task_id_t(), cpu->getInstrumentationId(), _instrumentationId);
	
	// The WorkerThread will iterate until its CPU status signals that there is
	// an ongoing shutdown and thus the thread must stop executing
	while (CPUActivation::checkCPUStatusTransitions(this) != CPU::shutting_down_status) {
		// Update the CPU since the thread may have migrated
		cpu = getComputePlace();
		assert(cpu != nullptr);
		instrumentationContext.updateComputePlace(cpu->getInstrumentationId());
		
		if (_task == nullptr) {
			_task = Scheduler::getReadyTask(cpu);
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
			PollingAPI::handleServices();
		}
	}
	
	// The thread should not have any task assigned at this point
	assert(_task == nullptr);
	
	Instrument::threadWillShutdown();
	
	Monitoring::shutdownThread();
	HardwareCounters::shutdownThread();
	
	WorkerThread *newThread = ThreadManager::getAnyIdleThread();
	if (newThread != nullptr) {
		newThread->resume(cpu, true);
	}
	
	ThreadManager::addShutdownThread(this);
}


void WorkerThread::handleTask(CPU *cpu)
{
	size_t NUMAId = cpu->getNumaNodeId();
	//MemoryPlace *targetPlace = cpu->getMemoryPlace(NUMAId);
	MemoryPlace *targetMemoryPlace = HardwareInfo::getMemoryPlace(nanos6_host_device, NUMAId);
	assert(targetMemoryPlace != nullptr);
	
	ExecutionWorkflow::executeTask(_task, cpu, targetMemoryPlace);
	
	_task = nullptr;
}


