/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
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
#include "hardware/HardwareInfo.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/threads/ExternalThread.hpp"
#include "lowlevel/threads/ExternalThreadGroup.hpp"
#include "scheduling/Scheduler.hpp"
#include "system/APICheck.hpp"
#include "system/RuntimeInfoEssentials.hpp"
#include "system/ompss/SpawnFunction.hpp"
#include "tasks/StreamManager.hpp"

#include <ClusterManager.hpp>
#include <DependencySystem.hpp>
#include <HardwareCounters.hpp>
#include <InstrumentInitAndShutdown.hpp>
#include <InstrumentThreadManagement.hpp>
#include <Monitoring.hpp>

static ExternalThread *mainThread = nullptr;

void nanos6_shutdown(void);

int nanos6_can_run_main(void)
{
	if (ClusterManager::isMasterNode()) {
		return true;
	} else {
		return false;
	}
}

void nanos6_register_completion_callback(void (*shutdown_callback)(void *), void *callback_args)
{
	assert(shutdown_callback != nullptr);
	ClusterManager::setShutdownCallback(shutdown_callback, callback_args);
}

void nanos6_preinit(void) {
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
	
	RuntimeInfoEssentials::initialize();
	HardwareInfo::initialize();
	ClusterManager::initialize();
	CPUManager::preinitialize();
	MemoryAllocator::initialize();
	Scheduler::initialize();
	ExternalThreadGroup::initialize();
	
	mainThread = new ExternalThread("main-thread");
	mainThread->preinitializeExternalThread();
	Instrument::initialize();
	
	HardwareCounters::initialize();
	Monitoring::initialize();
	
	mainThread->initializeExternalThread(/* already preinitialized */ false);
	
	// Register mainThread so that it will be automatically deleted
	// when shutting down Nanos6
	ExternalThreadGroup::registerExternalThread(mainThread);
	Instrument::threadHasResumed(mainThread->getInstrumentationId());
	
	ThreadManager::initialize();
	DependencySystem::initialize();
	LeaderThread::initialize();
	
	CPUManager::initialize();
}


void nanos6_init(void) {
	Instrument::threadWillSuspend(mainThread->getInstrumentationId());
	
	StreamManager::initialize();
}


void nanos6_shutdown(void) {
	Instrument::threadHasResumed(mainThread->getInstrumentationId());
	Instrument::threadWillShutdown();
	
	while (SpawnedFunctions::_pendingSpawnedFunctions > 0) {
		// Wait for spawned functions to fully end
	}
	
	StreamManager::shutdown();
	LeaderThread::shutdown();
	// Signal the shutdown to all CPUs and finalize threads
	CPUManager::shutdownPhase1();
	ThreadManager::shutdownPhase1();
	
	Instrument::shutdown();
	
	// Delete the worker threads
	// NOTE: AFTER Instrument::shutdown since it may need thread info!
	ThreadManager::shutdownPhase2();
	CPUManager::shutdownPhase2();
	
	// Delete all registered external threads, including mainThread
	ExternalThreadGroup::shutdown();
	
	Monitoring::shutdown();
	HardwareCounters::shutdown();
	
	Scheduler::shutdown();
	MemoryAllocator::shutdown();
	ClusterManager::shutdown();
	RuntimeInfoEssentials::shutdown();
}

