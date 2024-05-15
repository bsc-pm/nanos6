/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2024 Barcelona Supercomputing Center (BSC)
*/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <alloca.h>
#include <atomic>
#include <cassert>
#include <cstring>
#include <pthread.h>

#include "CPUManager.hpp"
#include "TaskFinalization.hpp"
#include "TaskFinalizationImplementation.hpp"
#include "ThreadManager.hpp"
#include "WorkerThread.hpp"
#include "dependencies/SymbolTranslation.hpp"
#include "hardware/device/directory/Directory.hpp"
#include "hardware/HardwareInfo.hpp"
#include "lowlevel/TurboSettings.hpp"
#include "scheduling/Scheduler.hpp"
#include "system/If0Task.hpp"
#include "system/TrackingPoints.hpp"
#include "tasks/LoopGenerator.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"

#include <DataAccessRegistration.hpp>
#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <InstrumentWorkerThread.hpp>

std::atomic<uint64_t> WorkerThread::_initializedThreads;

void WorkerThread::initialize()
{
	markAsCurrentWorkerThread();

	CPU *cpu = getComputePlace();
	assert(cpu != nullptr);

	Instrument::ThreadInstrumentationContext instrumentationContext(
		Instrument::task_id_t(),
		cpu->getInstrumentationId(),
		getInstrumentationId()
	);

	// Runtime Tracking Point - A thread is initializing
	TrackingPoints::threadInitialized(this, cpu);

	// At this point we're in the initial CPU
	// Perform warm-up phase if required
	if (TurboSettings::isWarmupEnabled())
		TurboSettings::warmupPhase();

	_initializedThreads.fetch_add(1, std::memory_order_relaxed);

	// This is needed for kernel-level threads to stop them after initialization
	synchronizeInitialization();

	if (TurboSettings::isWarmupEnabled()) {
		while (getInitializedThreads() < CPUManager::getAvailableCPUs())
			spinWait();
		spinWaitRelease();
	}
}

void WorkerThread::body()
{
	initialize();

	CPU *cpu = getComputePlace();
	assert(cpu != nullptr);

	Instrument::ThreadInstrumentationContext instrumentationContext(
		Instrument::task_id_t(), cpu->getInstrumentationId(), _instrumentationId
	);

	Instrument::workerThreadBegin();

	// There may be some CPUs in sponge mode. A CPU is in sponge mode when it is
	// available for the runtime system, but it does not execute any task on it to
	// avoid consuming its CPU time
	if (CPUManager::checkCPUStatusTransitions(this) != CPU::shutdown_status && CPUManager::isSpongeCPU(cpu)) {
		assert(cpu->isOwned());

		Instrument::enterSpongeMode();
		Instrument::workerAbsorbing();

		// Block the current thread on the CPU until the runtime finalization
		CPUManager::enterSpongeMode(cpu);

		assert(CPUManager::checkCPUStatusTransitions(this) == CPU::shutdown_status);

		Instrument::workerProgressing();
		Instrument::exitSpongeMode();
	}

	// The WorkerThread will iterate until its CPU status signals that there is
	// an ongoing shutdown and thus the thread must stop executing
	while (CPUManager::checkCPUStatusTransitions(this) != CPU::shutdown_status) {
		cpu = getComputePlace();
		assert(cpu != nullptr);

		// Update the CPU since the thread may have migrated
		instrumentationContext.updateComputePlace(cpu->getInstrumentationId());
		assert(_task == nullptr);

		Task *immediateSuccessor = cpu->getFirstSuccessor();
		if (immediateSuccessor) {
			// Draw a random number between 0.0 and 1.0
			float randomISValue = _ISDistribution(_ISGenerator);

			// Can only execute Immediate Successor if we accept replacement and probability allows us
			bool probabilityOfIS = (randomISValue < Scheduler::getImmediateSuccessorAlpha());
			if (!probabilityOfIS) {
				Scheduler::addReadyTask(
					immediateSuccessor,
					(immediateSuccessor->getDeviceType() == cpu->getType() ? cpu : nullptr),
					SIBLING_TASK_HINT);
			} else {
				// Check if the task needs to execute an onReady handler
				if (immediateSuccessor->handleOnready(this))
					_task = immediateSuccessor;

				// Otherwise, we have no IS and should just get a scheduler task
			}
			cpu->setFirstSuccessor(nullptr);
		}

		// No immediate successor, get a task
		if (_task == nullptr) {
			_task = Scheduler::getReadyTask(cpu, this);
		}

		if (_task != nullptr) {
			WorkerThread *assignedThread = _task->getThread();

			// A task already assigned to another thread
			if (assignedThread != nullptr) {
				_task = nullptr;

				// Runtime Tracking Point - The current thread will suspend
				TrackingPoints::threadWillSuspend(this, cpu);

				ThreadManager::addIdler(this);

				switchTo(assignedThread);
			} else {
				Instrument::workerThreadObtainedTask();

				if (_task->isIf0()) {
					// An if0 task executed outside of the implicit taskwait of its parent (i.e. not inline)
					Task *if0Task = _task;

					// This is needed, since otherwise the semantics would be that the if0Task task is being launched from within its own execution
					_task = nullptr;

					If0Task::executeNonInline(this, if0Task, cpu);
				} else {
					handleTask(cpu, true);
				}

				_task = nullptr;
			}
			CPUManager::checkIfMustReturnCPU(this);
		} else {
			// If no task is available, the CPUManager may want to idle this CPU
			CPUManager::executeCPUManagerPolicy(cpu, IDLE_CANDIDATE);
		}
		Instrument::workerThreadSpins();
	}

	// The thread should not have any task assigned at this point
	assert(_task == nullptr);

	Instrument::workerThreadEnd();

	// Runtime Tracking Point - The current thread is gonna shutdown
	TrackingPoints::threadWillShutdown();

	ThreadManager::addShutdownThread(this);
}

void WorkerThread::handleTask(CPU *cpu, bool)
{
	assert(_task != nullptr);
	assert(cpu != nullptr);

	Instrument::enterHandleTask();

	// Execute the task
	if (_task != nullptr) {
		executeTask(cpu);
	}

	Instrument::exitHandleTask();
}

void WorkerThread::executeTask(CPU *cpu)
{
	assert(_task != nullptr);
	assert(cpu != nullptr);

	nanos6_address_translation_entry_t stackTranslationTable[SymbolTranslation::MAX_STACK_SYMBOLS];

	const size_t NUMAId = cpu->getNumaNodeId();
	MemoryPlace *memoryPlace = HardwareInfo::getMemoryPlace(nanos6_host_device, NUMAId);
	assert(memoryPlace != nullptr);

	Instrument::task_id_t taskId = _task->getInstrumentationTaskId();
	Instrument::ThreadInstrumentationContext instrumentationContext(
		taskId, cpu->getInstrumentationId(), _instrumentationId
	);

	if (_task->hasCode()) {
		size_t tableSize = 0;
		nanos6_address_translation_entry_t *translationTable =
			SymbolTranslation::generateTranslationTable(
				_task, stackTranslationTable, tableSize
			);

		nanos6_task_info_t const *const taskInfo = _task->getTaskInfo();
		assert(taskInfo != nullptr);
		const int numSymbols = taskInfo->num_symbols;

		bool copiesReady = Directory::preTaskExecution(Directory::getHostAgent(), _task, translationTable, numSymbols);
		if(!copiesReady) {
			// Directory copies are not ready
			// The copies are queued, but we cannot execute this task yet
			// Free up all symbol translation
			if (tableSize > 0) {
				MemoryAllocator::free(translationTable, tableSize);
			}

			return;
		}

		SymbolTranslation::translateReductions(_task, cpu, translationTable, numSymbols);

		_task->setThread(this);
		_task->setMemoryPlace(memoryPlace);

		// Runtime Tracking Point - A task starts its execution
		TrackingPoints::taskIsExecuting(_task);

		// Run the task
		std::atomic_thread_fence(std::memory_order_acquire);
		_task->body(translationTable);
		std::atomic_thread_fence(std::memory_order_release);

		// Update the CPU since the thread may have migrated
		cpu = getComputePlace();
		instrumentationContext.updateComputePlace(cpu->getInstrumentationId());

		// Runtime Tracking Point - A task has completed its execution (user code)
		TrackingPoints::taskCompletedUserCode(_task);

		// Free up all symbol translation
		if (tableSize > 0) {
			MemoryAllocator::free(translationTable, tableSize);
		}
	} else {
		// Runtime Tracking Point - A task has completed its execution (user code)
		TrackingPoints::taskCompletedUserCode(_task);
	}

	DataAccessRegistration::combineTaskReductions(_task, cpu);

	if (_task->markAsFinished(cpu)) {
		DataAccessRegistration::unregisterTaskDataAccesses(
			_task, cpu, cpu->getDependencyData()
		);

		TaskFinalization::taskFinished(_task, cpu);
		if (_task->markAsReleased()) {
			TaskFinalization::disposeTask(_task);
		}
	}
}
