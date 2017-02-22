#include <iostream>

#include <assert.h>
#include <dlfcn.h>
#include <signal.h>

#include "LeaderThread.hpp"

#include <nanos6.h>
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/CPUManager.hpp"
#include "executors/threads/ThreadManagerPolicy.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "scheduling/Scheduler.hpp"
#include "system/ompss/SpawnFunction.hpp"
#include "hardware/HardwareInfo.hpp"

#include <InstrumentInitAndShutdown.hpp>


static std::atomic<int> shutdownDueToSignalNumber(0);

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
	HardwareInfo::initialize();
	Scheduler::initialize();
	ThreadManagerPolicy::initialize();
	CPUManager::preinitialize();
	Instrument::initialize();
	LeaderThread::initialize();
}


void nanos_init(void) {
	CPUManager::initialize();
	
	EnvironmentVariable<bool> handleSigInt("NANOS_HANDLE_SIGINT", 0);
	if (handleSigInt) {
		programSignal(SIGINT);
	}
	EnvironmentVariable<bool> handleSigTerm("NANOS_HANDLE_SIGTERM", 0);
	if (handleSigTerm) {
		programSignal(SIGTERM);
	}
	EnvironmentVariable<bool> handleSigQuit("NANOS_HANDLE_SIGQUIT", 0);
	if (handleSigQuit) {
		programSignal(SIGQUIT);
	}
	
	#ifndef NDEBUG
		programSignal(SIGABRT);
	#endif
}


void nanos_shutdown(void) {
	while (SpawnedFunctions::_pendingSpawnedFuncions > 0) {
		// Wait for spawned functions to fully end
	}
	
	LeaderThread::shutdown();
	Instrument::shutdown();
	
	if (shutdownDueToSignalNumber.load() != 0) {
		raise(shutdownDueToSignalNumber.load());
	}
	
	ThreadManager::shutdown();
	Scheduler::shutdown();
}

