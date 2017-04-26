#include "CPUThreadingModelData.hpp"
#include "executors/threads/CPUActivation.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"

#include <cassert>


std::atomic<long> CPUThreadingModelData::_shutdownThreads(0);
std::atomic<WorkerThread *> CPUThreadingModelData::_mainShutdownControllerThread(nullptr);


void CPUThreadingModelData::initialize(__attribute__((unused)) CPU *cpu)
{
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
