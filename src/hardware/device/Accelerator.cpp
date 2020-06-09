/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "Accelerator.hpp"

#include "executors/threads/TaskFinalization.hpp"
#include "hardware/HardwareInfo.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/TaskImplementation.hpp"

#include <DataAccessRegistration.hpp>
#include <Monitoring.hpp>

void Accelerator::runTask(Task *task)
{
	assert(task != nullptr);
	task->setComputePlace(_computePlace);
	task->setMemoryPlace(_memoryPlace);
	generateDeviceEvironment(task);

	preRunTask(task);
	setActiveDevice();
	task->body(nullptr);

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

	if(task->isIf0()) {
		Task *parent = task->getParent();
		assert(parent != nullptr);

		// Unlock parent that was waiting for this if0
		Scheduler::addReadyTask(parent, cpu, UNBLOCKED_TASK_HINT);

		// After adding a task, the CPUManager may want to un-idle CPUs
		CPUManager::executeCPUManagerPolicy(cpu, ADDED_TASKS, 1);
	}

	if (task->markAsFinished(cpu)) {
		DataAccessRegistration::unregisterTaskDataAccesses(
			task, cpu, hpDependencyData,
			task->getMemoryPlace(),
			/* from busy thread */ true);

		Monitoring::taskFinished(task);
		task->setComputePlace(nullptr);
		TaskFinalization::taskFinished(task, cpu, /* busy thread */ true);

		if (task->markAsReleased()) {
			TaskFinalization::disposeTask(task);
		}
	}
};
