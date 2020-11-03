/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2018-2020 Barcelona Supercomputing Center (BSC)
*/

#include "ExecutionWorkflow.hpp"
#include "dependencies/SymbolTranslation.hpp"
#include "executors/threads/TaskFinalization.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "hardware/places/MemoryPlace.hpp"
#include "system/ompss/MetricPoints.hpp"
#include "tasks/Task.hpp"
#include "tasks/Taskfor.hpp"
#include "tasks/TaskImplementation.hpp"

#include <DataAccessRegistration.hpp>
#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <InstrumentThreadManagement.hpp>


namespace ExecutionWorkflow {

	void executeTask(Task *task, ComputePlace *targetComputePlace, MemoryPlace *targetMemoryPlace)
	{
		nanos6_address_translation_entry_t stackTranslationTable[SymbolTranslation::MAX_STACK_SYMBOLS];
		CPU *cpu = (CPU *) targetComputePlace;
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		assert(currentThread != nullptr);
		assert(task != nullptr);

		task->setThread(currentThread);
		task->setMemoryPlace(targetMemoryPlace);

		Instrument::task_id_t taskId = task->getInstrumentationTaskId();
		Instrument::ThreadInstrumentationContext instrumentationContext(
			taskId,
			cpu->getInstrumentationId(),
			currentThread->getInstrumentationId()
		);

		bool taskHasCode = task->hasCode();
		if (taskHasCode) {
			size_t tableSize = 0;
			nanos6_address_translation_entry_t *translationTable =
				SymbolTranslation::generateTranslationTable(
					task, targetComputePlace,
					stackTranslationTable, tableSize);

			// Runtime Core Metric Point - A task starts its execution
			MetricPoints::taskIsExecuting(task);

			// Run the task
			std::atomic_thread_fence(std::memory_order_acquire);
			task->body(translationTable);
			std::atomic_thread_fence(std::memory_order_release);

			// Free up all symbol translation
			if (tableSize > 0)
				MemoryAllocator::free(translationTable, tableSize);

			// Update the CPU since the thread may have migrated
			cpu = currentThread->getComputePlace();
			instrumentationContext.updateComputePlace(cpu->getInstrumentationId());
		}

		// Runtime Core Metric Point - A task has completed its execution (user code)
		MetricPoints::taskCompletedUserCode(task);

		DataAccessRegistration::combineTaskReductions(task, cpu);

		if (task->markAsFinished(cpu)) {
			DataAccessRegistration::unregisterTaskDataAccesses(
				task, cpu, cpu->getDependencyData()
			);

			// Runtime Core Metric Point - A task has completely finished its execution
			MetricPoints::taskFinished(task);

			TaskFinalization::taskFinished(task, cpu);
			if (task->markAsReleased()) {
				TaskFinalization::disposeTask(task);
			}
		}
	}

	void setupTaskwaitWorkflow(Task *task, DataAccess *taskwaitFragment)
	{
		ComputePlace *computePlace = nullptr;
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		if (currentThread != nullptr) {
			computePlace = currentThread->getComputePlace();
		}

		CPUDependencyData hpDependencyData;
		DataAccessRegistration::releaseTaskwaitFragment(
			task, taskwaitFragment->getAccessRegion(),
			computePlace, hpDependencyData
		);
	}
}
