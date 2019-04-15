/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#include <assert.h>
#include <config.h>
#include <dlfcn.h>
#include <iostream>
#include <signal.h>

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

#include <ClusterManager.hpp>
#include <DependencySystem.hpp>
#include <HardwareCounters.hpp>
#include <InstrumentInitAndShutdown.hpp>
#include <InstrumentThreadManagement.hpp>
#include <Monitoring.hpp>
#include <WisdomManager.hpp>


static std::atomic<int> shutdownDueToSignalNumber(0);

static ExternalThread *mainThread = nullptr;

void nanos6_shutdown(void);


static void signalHandler(int signum)
{
	// SIGABRT needs special handling
	if (signum == SIGABRT) {
		Instrument::shutdown();
		return;
	}
	
	// For the rest, just set up the termination flag
	shutdownDueToSignalNumber.store(signum);
	nanos6_shutdown();
	
}


static void programSignal(int signum) {
	struct sigaction sa;
	sa.sa_handler = signalHandler;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_RESETHAND;
	
	int rc = sigaction(signum, &sa, nullptr);
	FatalErrorHandler::handle(rc, "Programming signal handler for signal number ", signum);
}

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
	MemoryAllocator::initialize();
	CPUManager::preinitialize();
	Scheduler::initialize();
	ExternalThreadGroup::initialize();
	
	mainThread = new ExternalThread("main-thread");
	mainThread->preinitializeExternalThread();
	Instrument::initialize();
	
	HardwareCounters::initialize();
	Monitoring::initialize();
	WisdomManager::initialize();
	
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
	EnvironmentVariable<bool> handleSigInt("NANOS6_HANDLE_SIGINT", 0);
	if (handleSigInt) {
		programSignal(SIGINT);
	}
	EnvironmentVariable<bool> handleSigTerm("NANOS6_HANDLE_SIGTERM", 0);
	if (handleSigTerm) {
		programSignal(SIGTERM);
	}
	EnvironmentVariable<bool> handleSigQuit("NANOS6_HANDLE_SIGQUIT", 0);
	if (handleSigQuit) {
		programSignal(SIGQUIT);
	}
	
	#ifndef NDEBUG
		programSignal(SIGABRT);
	#endif
	
	Instrument::threadWillSuspend(mainThread->getInstrumentationId());
}


void nanos6_shutdown(void) {
	Instrument::threadHasResumed(mainThread->getInstrumentationId());
	Instrument::threadWillShutdown();
	
	while (SpawnedFunctions::_pendingSpawnedFunctions > 0) {
		// Wait for spawned functions to fully end
	}
	
	LeaderThread::shutdown();
	ThreadManager::shutdown();
	
	Instrument::shutdown();
	
	// Delete all registered external threads, including mainThread
	ExternalThreadGroup::shutdown();
	
	if (shutdownDueToSignalNumber.load() != 0) {
		raise(shutdownDueToSignalNumber.load());
	}
	
	WisdomManager::shutdown();
	Monitoring::shutdown();
	HardwareCounters::shutdown();
	
	Scheduler::shutdown();
	MemoryAllocator::shutdown();
	ClusterManager::shutdown();
	RuntimeInfoEssentials::shutdown();
}

