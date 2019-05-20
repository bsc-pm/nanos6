/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018-2019 Barcelona Supercomputing Center (BSC)
*/

#include "ExecutionWorkflow.hpp"
#include "executors/threads/TaskFinalization.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "hardware/places/MemoryPlace.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"

#include <DataAccessRegistration.hpp>
#include <HardwareCounters.hpp>
#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentTaskExecution.hpp>
#include <InstrumentTaskStatus.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <InstrumentThreadManagement.hpp>
#include <Monitoring.hpp>


namespace ExecutionWorkflow {
	void executeTask(
		Task *task,
		ComputePlace *targetComputePlace,
		MemoryPlace *targetMemoryPlace
	) {
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		assert(currentThread != nullptr);
		CPU *cpu = (CPU *)targetComputePlace;
		
		task->setThread(currentThread);
		task->setMemoryPlace(targetMemoryPlace);
		
		Instrument::task_id_t taskId = task->getInstrumentationTaskId();
		Instrument::ThreadInstrumentationContext instrumentationContext(
			taskId,
			cpu->getInstrumentationId(),
			currentThread->getInstrumentationId()
		);
		
		if (task->hasCode()) {
			nanos6_address_translation_entry_t *translationTable = nullptr;
			
			nanos6_task_info_t const * const taskInfo = task->getTaskInfo();
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
			
			HardwareCounters::startTaskMonitoring(task);
			Monitoring::taskChangedStatus(task, executing_status, cpu);
			
			// Run the task
			std::atomic_thread_fence(std::memory_order_acquire);
			task->body(nullptr, translationTable);
			std::atomic_thread_fence(std::memory_order_release);
			
			// Update the CPU since the thread may have migrated
			cpu = currentThread->getComputePlace();
			instrumentationContext.updateComputePlace(cpu->getInstrumentationId());
			
			Monitoring::taskChangedStatus(task, runtime_status);
			Monitoring::taskCompletedUserCode(task, cpu);
			HardwareCounters::stopTaskMonitoring(task);
			
			Instrument::taskIsZombie(taskId);
			Instrument::endTask(taskId);
		} else {
			Monitoring::taskChangedStatus(task, runtime_status);
			Monitoring::taskCompletedUserCode(task, cpu);
			HardwareCounters::stopTaskMonitoring(task);
		}
		
		if (task->markAsFinished(cpu)) {
			DataAccessRegistration::unregisterTaskDataAccesses(
				task,
				cpu,
				cpu->getDependencyData()
			);
			
			Monitoring::taskFinished(task);
			HardwareCounters::taskFinished(task);
			
			if (task->markAsReleased()) {
				TaskFinalization::disposeOrUnblockTask(task, cpu);
			}
		}
	}
	
	void setupTaskwaitWorkflow(
		Task *task,
		DataAccess *taskwaitFragment
	) {
		ComputePlace *computePlace = nullptr;
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		if (currentThread != nullptr) {
			computePlace = currentThread->getComputePlace();
		}
		
		CPUDependencyData hpDependencyData;
		DataAccessRegistration::releaseTaskwaitFragment(
			task,
			taskwaitFragment->getAccessRegion(),
			computePlace,
			hpDependencyData
		);
	}
}
