/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <iostream>

#include <assert.h>
#include <dlfcn.h>
#include <signal.h>

#include "LeaderThread.hpp"
#include "MemoryAllocator.hpp"

#include <nanos6.h>
#include <nanos6/bootstrap.h>

#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/CPUManager.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/threads/ExternalThread.hpp"
#include "scheduling/Scheduler.hpp"
#include "system/APICheck.hpp"
#include "system/RuntimeInfoEssentials.hpp"
#include "system/ompss/SpawnFunction.hpp"
#include "hardware/HardwareInfo.hpp"

#include <DependencySystem.hpp>
#include <InstrumentInitAndShutdown.hpp>
#include <InstrumentThreadManagement.hpp>

#include <config.h>

#ifdef USE_CUDA
#include "hardware/cuda/CUDAManager.hpp"
#endif

#ifdef USE_CLUSTER
#include "cluster/ClusterManager.hpp"
#endif


static std::atomic<int> shutdownDueToSignalNumber(0);

static ExternalThread *mainThread = nullptr;

void nanos_shutdown(void);


static void signalHandler(int signum)
{
	// SIGABRT needs special handling
	if (signum == SIGABRT) {
		Instrument::shutdown();
		return;
	}
	
	// For the rest, just set up the termination flag
	shutdownDueToSignalNumber.store(signum);
	nanos_shutdown();
	
}


static void programSignal(int signum) {
	struct sigaction sa;
	sa.sa_handler = signalHandler;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_RESETHAND;
	
	int rc = sigaction(signum, &sa, nullptr);
	FatalErrorHandler::handle(rc, "Programming signal handler for signal number ", signum);
}


void nanos_preinit(void) {
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
	MemoryAllocator::initialize();
	CPUManager::preinitialize();
	Scheduler::initialize();
	
	mainThread = new ExternalThread("main-thread");
	mainThread->preinitializeExternalThread();
	Instrument::initialize();
	mainThread->initializeExternalThread(/* already preinitialized */ false);
	Instrument::threadHasResumed(mainThread->getInstrumentationId());
	
	DependencySystem::initialize();
	LeaderThread::initialize();
	
	#ifdef USE_CUDA
	CUDAManager::initialize();
	#endif
	
	#ifdef USE_CLUSTER
	ClusterManager::initialize();
	#endif
}


void nanos_init(void) {
	CPUManager::initialize();
	
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


void nanos_shutdown(void) {
	Instrument::threadHasResumed(mainThread->getInstrumentationId());
	Instrument::threadWillShutdown();
	
	while (SpawnedFunctions::_pendingSpawnedFunctions > 0) {
		// Wait for spawned functions to fully end
	}
	
	#ifdef USE_CLUSTER
	ClusterManager::shutdown();
	#endif
	
	#ifdef USE_CUDA
	CUDAManager::shutdown();
	#endif
	
	LeaderThread::shutdown();
	ThreadManager::shutdown();
	
	Instrument::shutdown();
	delete mainThread;
	
	if (shutdownDueToSignalNumber.load() != 0) {
		raise(shutdownDueToSignalNumber.load());
	}
	
	Scheduler::shutdown();
	MemoryAllocator::shutdown();
	RuntimeInfoEssentials::shutdown();
}

