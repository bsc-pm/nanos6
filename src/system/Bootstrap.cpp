/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2023 Barcelona Supercomputing Center (BSC)
*/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <assert.h>
#include <config.h>
#include <dlfcn.h>
#include <iostream>

#include <nanos6.h>
#include <nanos6/bootstrap.h>

#include "LeaderThread.hpp"
#include "MemoryAllocator.hpp"
#include "executors/threads/CPUManager.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "hardware/device/directory/Directory.hpp"
#include "hardware/HardwareInfo.hpp"
#include "hardware-counters/HardwareCounters.hpp"
#include "lowlevel/TurboSettings.hpp"
#include "lowlevel/threads/ExternalThread.hpp"
#include "lowlevel/threads/ExternalThreadGroup.hpp"
#include "memory/numa/NUMAManager.hpp"
#include "monitoring/Monitoring.hpp"
#include "scheduling/Scheduler.hpp"
#include "support/config/ConfigCentral.hpp"
#include "support/config/ConfigChecker.hpp"
#include "system/APICheck.hpp"
#include "system/RuntimeInfoEssentials.hpp"
#include "system/SpawnFunction.hpp"
#include "system/Throttle.hpp"

#include "tasks/StreamManager.hpp"

#include <DependencySystem.hpp>
#include <InstrumentInitAndShutdown.hpp>
#include <InstrumentThreadManagement.hpp>
#include <InstrumentMainThread.hpp>

static ExternalThread *mainThread = nullptr;

void nanos6_shutdown(void);

int nanos6_can_run_main(void)
{
	return true;
}

void nanos6_register_completion_callback(void (*)(void *), void *)
{
}

void nanos6_preinit(void)
{
	if (!nanos6_api_has_been_checked_successfully()) {
		int *_nanos6_exit_with_error_ptr = (int *) dlsym(nullptr, "_nanos6_exit_with_error");
		if (_nanos6_exit_with_error_ptr != nullptr) {
			*_nanos6_exit_with_error_ptr = 1;
		}

		FatalErrorHandler::failIf(
			!nanos6_api_has_been_checked_successfully(),
			"this executable was compiled for a different Nanos6 version. Please recompile and link it."
		);
	}

	// Initialize all runtime options if needed
	ConfigCentral::initializeOptionsIfNeeded();

	// Begin the main thread just after the configuration is ready
	Instrument::mainThreadBegin();

	// Enable special flags for turbo mode
	TurboSettings::initialize();

	RuntimeInfoEssentials::initialize();

	// Pre-initialize Hardware Counters and Monitoring before hardware
	HardwareCounters::preinitialize();
	Monitoring::preinitialize();
	Directory::initialize();
	HardwareInfo::initialize();

	CPUManager::preinitialize();

	// Finish Hardware counters and Monitoring initialization after CPUManager
	HardwareCounters::initialize();
	Monitoring::initialize();
	MemoryAllocator::initialize();
	NUMAManager::initialize();
	Scheduler::initialize();
	Throttle::initialize();
	ExternalThreadGroup::initialize();

	Instrument::initialize();

	// Initialize device services after initializing scheduler and instrumentation
	HardwareInfo::initializeDeviceServices();

	mainThread = new ExternalThread("main-thread");
	mainThread->preinitializeExternalThread();
	mainThread->initializeExternalThread(/* already preinitialized */ false);

	// Register mainThread so that it will be automatically deleted
	// when shutting down Nanos6
	ExternalThreadGroup::registerExternalThread(mainThread);
	Instrument::threadHasResumed(mainThread->getInstrumentationId());

	ThreadManager::initialize();
	DependencySystem::initialize();

	// Retrieve the virtual CPU for the leader thread
	CPU *leaderThreadCPU = CPUManager::getLeaderThreadCPU();
	assert(leaderThreadCPU != nullptr);

	LeaderThread::initialize(leaderThreadCPU);

	CPUManager::initialize();
	Instrument::preinitFinished();

	// Assert config conditions if any
	ConfigChecker::assertConditions();
}


void nanos6_init(void)
{
	Instrument::threadWillSuspend(mainThread->getInstrumentationId());

	StreamManager::initialize();

	// The thread will be paused in the loader waiting in a condition
	// variable, but not via the instrumented suspend() method. So we
	// add an extra pause event here, and resume in shutdown.
	Instrument::pthreadPause();
}


void nanos6_shutdown(void)
{
	Instrument::pthreadResume();
	Instrument::threadHasResumed(mainThread->getInstrumentationId());
	Instrument::threadWillShutdown(mainThread->getInstrumentationId());

	while (SpawnFunction::_pendingSpawnedFunctions > 0) {
		// Wait for spawned functions to fully end
	}

	NUMAManager::shutdown();
	StreamManager::shutdown();
	LeaderThread::shutdown();

	// Shutdown device services before CPU and thread managers
	HardwareInfo::shutdownDeviceServices();

	// Shutdown throttle service before CPUs are stopped
	Throttle::shutdown();

	// Signal the shutdown to all CPUs and finalize threads
	CPUManager::shutdownPhase1();
	ThreadManager::shutdownPhase1();

	Instrument::shutdown();

	// Delete spawned functions task infos
	SpawnFunction::shutdown();

	// Delete the worker threads
	// NOTE: AFTER Instrument::shutdown since it may need thread info!
	ThreadManager::shutdownPhase2();
	CPUManager::shutdownPhase2();

	// Delete all registered external threads, including mainThread
	ExternalThreadGroup::shutdown();

	Monitoring::shutdown();
	HardwareCounters::shutdown();

	HardwareInfo::shutdown();
	Scheduler::shutdown();

	MemoryAllocator::shutdown();
	RuntimeInfoEssentials::shutdown();
	TurboSettings::shutdown();

	Instrument::mainThreadEnd();
}
