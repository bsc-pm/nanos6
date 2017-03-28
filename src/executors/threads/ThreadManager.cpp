#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cassert>
#include <list>

#include <pthread.h>
#include <unistd.h>

#include <sys/syscall.h>

#include "CPUActivation.hpp"
#include "CPUManager.hpp"
#include "ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"

#include <InstrumentThreadManagement.hpp>


std::atomic<bool> ThreadManager::_mustExit(false);
SpinLock ThreadManager::_idleThreadsLock;
std::deque<WorkerThread *> ThreadManager::_idleThreads;
std::atomic<long> ThreadManager::_totalThreads(0);
std::atomic<long> ThreadManager::_shutdownThreads(0);
std::atomic<WorkerThread *> ThreadManager::_mainShutdownControllerThread(nullptr);


void ThreadManager::initializeThread(CPU *cpu)
{
	assert(cpu != nullptr);
	assert(_shutdownThreads == 0);
	
	WorkerThread *thread = new WorkerThread(cpu);
	thread->_cpuToBeResumedOn = cpu;
	thread->resume();
	_totalThreads++;
}


void ThreadManager::threadStartup(WorkerThread *currentThread)
{
	assert(currentThread != nullptr);
	assert(currentThread->_cpu != nullptr);
	
	WorkerThread::_currentWorkerThread = currentThread;
	
	// Initialize the CPU status if necessary before the thread has a chance to check the shutdown signal
	CPUActivation::threadInitialization(currentThread);
	
	currentThread->setInstrumentationId(Instrument::createdThread());
	
	// The thread suspends itself after initialization, since the "activator" is the one will unblock it when needed
	currentThread->suspend();
	
	// Update the CPU since the thread may have migrated while blocked (or during pre-signaling)
	assert(currentThread->_cpuToBeResumedOn != nullptr);
	currentThread->_cpu = currentThread->_cpuToBeResumedOn;
	
	Instrument::threadHasResumed(currentThread->_instrumentationId, currentThread->_cpu->getInstrumentationId());
	
#ifndef NDEBUG
	currentThread->_cpuToBeResumedOn = nullptr;
#endif
}


void ThreadManager::shutdown()
{
	_mustExit = true;
	long shutdownThreads = _totalThreads;
	_shutdownThreads = shutdownThreads;
	
	// Attempt to wake up all (enabled) CPUs so that they start shutting down the threads
	std::vector<CPU *> cpus = CPUManager::getCPUListReference();
	std::deque<CPU *> participatingCPUs;
	for (CPU *cpu : cpus) {
		// Sanity check
		assert(_totalThreads == shutdownThreads);
		assert(_shutdownThreads <= shutdownThreads);
		
		if ((cpu != nullptr) && CPUActivation::acceptsWork(cpu)) {
			// Wait for the CPU to be started
			while (CPUActivation::isBeingInitialized(cpu)) {
				sched_yield();
			}
			
			WorkerThread *idleThread = getIdleThread(cpu, true);
			// Threads can be lagging behind (not in the idle queue yet), but we do need at least one.
			// On the other hand, the ones that have already started the shutdown can actually deplete
			// the rest of the idle threads.
			while ((idleThread == nullptr) && (_shutdownThreads > 0)) {
				sched_yield();
				idleThread = getIdleThread(cpu, true);
			}
			
			if (idleThread != nullptr) {
				// Set up the CPU shutdown controller thread
				assert(cpu->_shutdownControlerThread == nullptr);
				cpu->_shutdownControlerThread = idleThread;
				
				// Set up the main shutdown controller thread
				if (_mainShutdownControllerThread == nullptr) {
					_mainShutdownControllerThread = idleThread;
				}
				
				// Migrate the thread if necessary
				idleThread->_cpuToBeResumedOn = cpu;
				if (idleThread->_cpu != cpu) {
					cpu->bindThread(idleThread->_tid);
				}
				
				idleThread->signalShutdown();
				
				// Resume the thread
				idleThread->resume();
				
				// Place them in reverse order so the last one we get afterwards is the main shutdown controller
				participatingCPUs.push_front(cpu);
			}
		}
	}
	
	assert(_mainShutdownControllerThread != nullptr);
	
	// At this point we have woken as many threads as active CPUs. They perform the
	// shutdown collectively. The number can actually be smaller than activeCPUs.size().
	// The reason is that as soon as one starts the shutdown procedure, it will start
	// collecting other threads. That is, it will be compeing to get idle threads too.
	// However, there will be at least one of them, the main shutdown controller, and it
	// will be the last controller in "activeCPUs".
	
	// Join all the shutdown controller threads
	for (auto cpu : participatingCPUs) {
		// Sanity check
		assert(_totalThreads == shutdownThreads);
		assert(_shutdownThreads <= shutdownThreads);
		
		WorkerThread *shutdownControllerThread = cpu->_shutdownControlerThread;
		assert(shutdownControllerThread != nullptr);
		
		int rc = pthread_join(shutdownControllerThread->_pthread, nullptr);
		FatalErrorHandler::handle(rc, " during shutdown when joining pthread ", shutdownControllerThread->_pthread);
	}
	
	// Sanity check
	assert(_totalThreads == shutdownThreads);
	assert(_shutdownThreads == 0);
}


void ThreadManager::threadShutdownSequence(WorkerThread *currentThread)
{
	CPU *cpu = currentThread->_cpu;
	assert(cpu != nullptr);
	assert(WorkerThread::getCurrentWorkerThread() == currentThread);
	
	if (cpu->_shutdownControlerThread == currentThread) {
		// This thread is the shutdown controller (of the CPU)
		
		bool isMainController = (_mainShutdownControllerThread == currentThread);
		
		// Keep processing threads
		bool done = false;
		while (!done) {
			// Find next to wake up
			WorkerThread *next = getIdleThread(cpu, true);
			
			if (next != nullptr) {
				assert(next->getTask() == nullptr);
				
				next->signalShutdown();
				
				// Migrate the thread if necessary
				assert(next->_cpuToBeResumedOn == nullptr);
				next->_cpuToBeResumedOn = cpu;
				if (next->_cpu != cpu) {
					cpu->bindThread(next->_tid);
				}
				
				// Resume the thread
				next->resume();
				
				int rc = pthread_join(next->_pthread, nullptr);
				FatalErrorHandler::handle(rc, " during shutdown when joining pthread ", next->_pthread, " from pthread ", currentThread->_pthread, " in CPU ", cpu->_systemCPUId);
			} else {
				// No more idle threads (for the moment)
				if (!isMainController) {
					// Let the main shutdown controller handle any thread that may be lagging (did not enter the idle queue yet)
					done = true;
				} else if (_shutdownThreads == 1) {
					// This is the main shutdown controller and is also the last (worker) thread
					assert(isMainController);
					done = true;
				}
			}
		}
	}
	
	// Exit the current thread
	_shutdownThreads--;
	currentThread->exit();
}

