/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "Accelerator.hpp"
#include "dependencies/SymbolTranslation.hpp"
#include "executors/threads/TaskFinalization.hpp"
#include "hardware/HardwareInfo.hpp"
#include "scheduling/Scheduler.hpp"
#include "system/TrackingPoints.hpp"
#include "tasks/TaskImplementation.hpp"

#include <DataAccessRegistration.hpp>

void Accelerator::runTask(Task *task)
{
	nanos6_address_translation_entry_t stackTranslationTable[SymbolTranslation::MAX_STACK_SYMBOLS];

	assert(task != nullptr);
	task->setComputePlace(_computePlace);
	task->setMemoryPlace(_memoryPlace);

	setActiveDevice();
	generateDeviceEvironment(task);
	preRunTask(task);

	size_t tableSize = 0;
	nanos6_address_translation_entry_t *translationTable =
		SymbolTranslation::generateTranslationTable(
			task, _computePlace, stackTranslationTable,
			tableSize);

	task->body(translationTable);

	if (tableSize > 0)
		MemoryAllocator::free(translationTable, tableSize);

	postRunTask(task);
}

void Accelerator::finishTask(Task *task)
{
	finishTaskCleanup(task);

	WorkerThread *currThread = WorkerThread::getCurrentWorkerThread();

	CPU *cpu = nullptr;
	if (currThread != nullptr)
		cpu = currThread->getComputePlace();

	CPUDependencyData localDependencyData;
	CPUDependencyData &hpDependencyData = (cpu != nullptr) ? cpu->getDependencyData() : localDependencyData;

	if (task->isIf0()) {
		Task *parent = task->getParent();
		assert(parent != nullptr);

		// Unlock parent that was waiting for this if0
		Scheduler::addReadyTask(parent, cpu, UNBLOCKED_TASK_HINT);
	}

	if (task->markAsFinished(cpu)) {
		DataAccessRegistration::unregisterTaskDataAccesses(
			task, cpu, hpDependencyData,
			task->getMemoryPlace(),
			/* from busy thread */ true);

		TaskFinalization::taskFinished(task, cpu, /* busy thread */ true);

		if (task->markAsReleased()) {
			TaskFinalization::disposeTask(task);
		}
	}
}

void Accelerator::initializeService()
{
	// Spawn service function
	SpawnFunction::spawnFunction(
		serviceFunction, this,
		serviceCompleted, this,
		"Device service", false
	);
}

void Accelerator::shutdownService()
{
	// Notify the service to stop
	_stopService = true;

	// Wait until the service completes
	while (!_finishedService);
}

void Accelerator::serviceFunction(void *data)
{
	Accelerator *accel = (Accelerator *) data;
	assert(accel != nullptr);

	// Execute the service loop
	accel->acceleratorServiceLoop();
}

void Accelerator::serviceCompleted(void *data)
{
	Accelerator *accel = (Accelerator *) data;
	assert(accel != nullptr);
	assert(accel->_stopService);

	// Mark the service as completed
	accel->_finishedService = true;
}
