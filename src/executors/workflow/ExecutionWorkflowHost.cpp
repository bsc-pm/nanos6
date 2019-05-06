/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018-2019 Barcelona Supercomputing Center (BSC)
*/

#include "ExecutionWorkflowHost.hpp"
#include "executors/threads/TaskFinalization.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "hardware/places/MemoryPlace.hpp"
#include "tasks/Task.hpp"

#include <DataAccessRegistration.hpp>
#include <HardwareCounters.hpp>
#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentLogMessage.hpp>
#include <InstrumentTaskExecution.hpp>
#include <InstrumentTaskStatus.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <InstrumentThreadManagement.hpp>
#include <Monitoring.hpp>


namespace ExecutionWorkflow {
	
	void HostExecutionStep::start()
	{
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		CPU *cpu = (currentThread == nullptr) ?
			nullptr : currentThread->getComputePlace();
		
		//! We are trying to start the execution of the Task from within
		//! something that is not a WorkerThread, or it does not have 
		//! a CPU or the task assigned to it.
		//!
		//! This will happen once the last DataCopyStep finishes and
		//! releases the ExecutionStep.
		//!
		//! In that case we need to add the Task back for scheduling.
		if ((currentThread == nullptr) || (cpu == nullptr) ||
		   	(currentThread->getTask() == nullptr)
		) {
			_task->setExecutionStep(this);
			
			ComputePlace *idleComputePlace =
				Scheduler::addReadyTask(
					_task,
					nullptr,
					SchedulerInterface::BUSY_COMPUTE_PLACE_TASK_HINT
				);
			
			if (idleComputePlace != nullptr) {
				ThreadManager::resumeIdle((CPU *) idleComputePlace);
			}
			
			return;
		}
		
		_task->setThread(currentThread);
		Instrument::task_id_t taskId = _task->getInstrumentationTaskId();
		
		Instrument::ThreadInstrumentationContext instrumentationContext(
			taskId,
			cpu->getInstrumentationId(),
			currentThread->getInstrumentationId()
		);
		
		if (_task->hasCode()) {
			nanos6_address_translation_entry_t *translationTable = nullptr;
			
			nanos6_task_info_t const * const taskInfo = _task->getTaskInfo();
			if (taskInfo->num_symbols >= 0) {
				translationTable = (nanos6_address_translation_entry_t *)
						alloca(
							sizeof(nanos6_address_translation_entry_t)
							* taskInfo->num_symbols
						);
				
				for (int index = 0; index < taskInfo->num_symbols; index++) {
					translationTable[index] = {0, 0};
				}
			}
			
			Instrument::startTask(taskId);
			Instrument::taskIsExecuting(taskId);
			
			HardwareCounters::startTaskMonitoring(_task);
			Monitoring::taskChangedStatus(_task, executing_status, cpu);
			
			// Run the task
			std::atomic_thread_fence(std::memory_order_acquire);
			_task->body(nullptr, translationTable);
			std::atomic_thread_fence(std::memory_order_release);
			
			// Update the CPU since the thread may have migrated
			cpu = currentThread->getComputePlace();
			instrumentationContext.updateComputePlace(cpu->getInstrumentationId());
			
			Monitoring::taskChangedStatus(_task, runtime_status);
			Monitoring::taskCompletedUserCode(_task, cpu);
			HardwareCounters::stopTaskMonitoring(_task);
			
			Instrument::taskIsZombie(taskId);
			Instrument::endTask(taskId);
		} else {
			Monitoring::taskChangedStatus(_task, runtime_status);
			Monitoring::taskCompletedUserCode(_task, cpu);
			HardwareCounters::stopTaskMonitoring(_task);
		}
		
		//! Release the subsequent steps.
		releaseSuccessors();
		delete this;
	}
};
