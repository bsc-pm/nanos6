/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "CPUThreadingModelData.hpp"
#include "executors/threads/CPUActivation.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "system/RuntimeInfo.hpp"

#include <cassert>


std::atomic<long> CPUThreadingModelData::_shutdownThreads(0);
std::atomic<WorkerThread *> CPUThreadingModelData::_mainShutdownControllerThread(nullptr);

EnvironmentVariable<StringifiedMemorySize> CPUThreadingModelData::_defaultThreadStackSize("NANOS6_STACK_SIZE", 8 * 1024 * 1024);


void CPUThreadingModelData::initialize(__attribute__((unused)) CPU *cpu)
{
	static std::atomic<bool> firstTime(true);
	bool expect = true;
	bool worked = firstTime.compare_exchange_strong(expect, false);
	if (worked) {
		RuntimeInfo::addEntry("threading_model", "Threading Model", "pthreads");
		RuntimeInfo::addEntry("stack_size", "Stack Size", getDefaultStackSize());
	}
}


void CPUThreadingModelData::shutdownPhase1(CPU *cpu)
{
	if (_mainShutdownControllerThread == nullptr) {
		_shutdownThreads.store(ThreadManager::_totalThreads, std::memory_order_seq_cst);
	}
	
	// Wait for the CPU to be started
	while (CPUActivation::isBeingInitialized(cpu)) {
		sched_yield();
	}
	
	WorkerThread *idleThread = ThreadManager::getIdleThread(cpu, true);
	// Threads can be lagging behind (not in the idle queue yet), but we do need at least one.
	// On the other hand, the ones that have already started the shutdown can actually deplete
	// the rest of the idle threads.
	while ((idleThread == nullptr) && (_shutdownThreads > 0)) {
		sched_yield();
		idleThread = ThreadManager::getIdleThread(cpu, true);
	}
	
	if (idleThread != nullptr) {
		// Set up the CPU shutdown controller thread
		assert(_shutdownControllerThread == nullptr);
		_shutdownControllerThread = idleThread;
		
		// Set up the main shutdown controller thread
		if (_mainShutdownControllerThread == nullptr) {
			_mainShutdownControllerThread = idleThread;
		}
		
		idleThread->signalShutdown();
		
		// Resume the thread
		idleThread->resume(cpu, true);
	}
	
}


void CPUThreadingModelData::shutdownPhase2(__attribute__((unused)) CPU *cpu)
{
	if (_shutdownControllerThread.load() != nullptr) {
		_shutdownControllerThread.load()->join();
	} else {
		// The threads may have been exhausted before the CPU got a chance to get a shutdown controller assigned
	}
}
