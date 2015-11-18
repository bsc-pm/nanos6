#include <iostream>

#include <assert.h>
#include <dlfcn.h>

#include "LeaderThread.hpp"

#include "api/nanos6_rt_interface.h"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/ThreadManagerPolicy.hpp"
#include "scheduling/Scheduler.hpp"

#include <InstrumentInitAndShutdown.hpp>


void nanos_preinit(void) {
	Scheduler::initialize();
	ThreadManagerPolicy::initialize();
	Instrument::initialize();
}

void nanos_init(void) {
	ThreadManager::initialize();
}

void nanos_wait_until_shutdown(void) {
	LeaderThread::maintenanceLoop();
	Instrument::shutdown();
	ThreadManager::shutdown();
}

void nanos_notify_ready_for_shutdown(void) {
	LeaderThread::notifyMainExit();
}

