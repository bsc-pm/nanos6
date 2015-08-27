#ifndef THREAD_MANAGER_HPP
#define THREAD_MANAGER_HPP


#include <algorithm>
#include <atomic>
#include <cassert>
#include <set>
#include <stack>
#include <vector>

#include <pthread.h>
#include <unistd.h>

#include "hardware/places/HardwarePlace.hpp"
#include "lowlevel/SpinLock.hpp"

#include "CPU.hpp"
// #include "CPUStatusListener.hpp"
#include "WorkerThread.hpp"


class ThreadManagerDebuggingInterface;


class ThreadManager {
	//! \brief indicates if the runtime is shutting down
	static std::atomic<bool> _mustExit;
	
	//! \brief CPU mask of the process
	static cpu_set_t _processCPUMask;
	
	//! \brief per-CPU data indexed by system CPU identifier
	static std::vector<std::atomic<CPU *>> _cpus;
	
	//! \brief numer of initialized CPUs
	static std::atomic<long> _totalCPUs;
	
	static SpinLock _idleCPUsLock;
	
	//! \brief CPUs that are currently idle
	static std::deque<CPU *> _idleCPUs;
	
	//! \brief retrieve an idle WorkerThread that is pinned to a given CPU
	//! NOTE: This method should be called with the CPU status lock held
	//!
	//! \param inout cpu the CPU on which to get an idle thread
	//!
	//! \returns an idle WorkerThread pinned to the requested CPU or nullptr (if none left, or the CPU is disabled)
	static inline WorkerThread *getIdleThread(CPU *cpu);
	
	//! \brief create or recycle a WorkerThread that is pinned to a given CPU
	//! NOTE: This method should be called with the CPU status lock held
	//!
	//! \param inout cpu the CPU on which to get a new or idle thread
	//!
	//! \returns a WorkerThread pinned to the requested CPU or nullptr if the CPU is disabled
	static inline WorkerThread *getNewOrIdleThread(CPU *cpu);
	
	//! \brief get a ready WorkerThread that is pinned to a given CPU
	//! NOTE: This method should be called with the CPU status lock held
	//!
	//! \param inout cpu the CPU on which to get a ready thread
	//!
	//! \returns a WorkerThread pinned to the requested CPU that is ready (previously blocked)
	static inline WorkerThread *getReadyThread(CPU *cpu);
	
	//! \brief suspend the currently running thread and replace it by another (if given)
	//! NOTE: This method should be called with the CPU status lock held
	//!
	//! \param in currentThread a thread that is currently running and that must be stopped
	//! \param in replacementThread a thread that is currently suspended and that must take the place of the currentThread or nullptr
	//! \param inout cpu the CPU that is running the currentThread and that must run the replacementThread
	static inline void switchThreads(WorkerThread *currentThread, WorkerThread *replacementThread, CPU *cpu);
	
	static inline void linkIdleCPU(CPU *cpu);
	static inline void unlinkIdleCPU(CPU *cpu);
	static inline CPU *getIdleCPU();
	
	
public:
	static void initialize();
	
	static void shutdown();
	
	
	//! \brief get or create the CPU object assigned to a given numerical system CPU identifier
	static inline CPU *getCPU(size_t systemCPUId);
	
	//! \brief get the maximum number of CPUs that will be used
	static inline long getTotalCPUs();
	
	//! \brief suspend the currently running thread due to idleness and potentially switch to another
	//! Threads suspended with this call must only be woken up through a call to resumeAnyIdle
	//!
	//! \param in currentThread a thread that is currently running and that must be stopped
	static inline void yieldIdler(WorkerThread *currentThread);
	
	//! \brief suspend the currently running thread due to a blocking condition and potentially switch to another
	//! Threads suspended with this call can only be woken up through a call to threadBecomesReady. Therefore, the
	//! entity responsible for the call to this method should also keep track of the fact that the thread is 
	//! blocked and invoke the threadBecomesReady method at a proper time.
	//!
	//! \param in currentThread a thread that is currently running and that must be stopped
	static inline void suspendForBlocking(WorkerThread *currentThread);
	
	//! \brief resume an idle thread on a given CPU
	//!
	//! \param in idleCPU the CPU on which to resume an idle thread
	//! \param in inInitialization true if it should not enforce assertions that are not valid during initialization
	static inline void resumeIdle(CPU *idleCPU, bool inInitialization=false);
	
	//! \brief attempt to activate an idle CPU
	//!
	//! \param in preferredHardwarePlace a hint on a desirable area of the hardware
	static inline void resumeAnyIdle(HardwarePlace *preferredHardwarePlace);
	
	//! \brief indicate that a previously ready thread has become ready
	//!
	//! \param in readyThread a thready that was blocked and now is ready to be executed again
	static inline void threadBecomesReady(WorkerThread *readyThread);
	
	
	//! \brief returns true if the thread must shut down
	static inline bool mustExit();
	
	//! \brief set up the information related to the currently running thread
	//!
	//! \param in currentThread a thread that is currently running and that has just begun its execution
	static void threadStartup(WorkerThread *currentThread);
	
	//! \brief exit the currently running thread and wake up the next one assigned to the same CPU (so that it can do the same)
	//!
	//! \param in currentThread a thread that is currently running and that must exit
	static void exitAndWakeUpNext(WorkerThread *currentThread);
	
	friend class ThreadManagerDebuggingInterface;
};


CPU *ThreadManager::getCPU(size_t systemCPUId)
{
	assert(systemCPUId < _cpus.size());
	
	CPU *cpu = _cpus[systemCPUId];
	if (cpu == nullptr) {
		CPU *newCPU = new CPU(systemCPUId, /* INVALID VALUE */ ~0UL);
		bool success = _cpus[systemCPUId].compare_exchange_strong(cpu, newCPU);
		if (!success) {
			// Another thread already did it
			delete newCPU;
			cpu = _cpus[systemCPUId];
		} else {
			newCPU->_virtualCPUId = _totalCPUs++;
			cpu = newCPU;
		}
	}
	
	return cpu;
}


inline long ThreadManager::getTotalCPUs()
{
	return _totalCPUs;
}


//
// Getting / Returning worker threads
//


inline WorkerThread *ThreadManager::getIdleThread(CPU *cpu)
{
	assert(cpu != nullptr);
	assert(cpu->_statusLock.isLockedByThisThread());
	
	if (!cpu->_idleThreads.empty()) {
		WorkerThread *idleThread = cpu->_idleThreads.front();
		cpu->_idleThreads.pop_front();
		
		return idleThread;
	}
	
	return nullptr;
}


inline WorkerThread *ThreadManager::getNewOrIdleThread(CPU *cpu)
{
	assert(cpu != nullptr);
	assert(cpu->_statusLock.isLockedByThisThread());
	
	WorkerThread *idleThread = getIdleThread(cpu);
	if (idleThread != nullptr) {
		return idleThread;
	}
	
	return new WorkerThread(cpu);
}

inline WorkerThread *ThreadManager::getReadyThread(CPU *cpu)
{
	assert(cpu != nullptr);
	assert(cpu->_statusLock.isLockedByThisThread());
	WorkerThread *readyThread = nullptr;
	
	if (!cpu->_readyThreads.empty()) {
		readyThread = cpu->_readyThreads.front();
		cpu->_readyThreads.pop_front();
		assert(readyThread->_cpu == cpu);
	}
	
	return readyThread;
}


inline void ThreadManager::linkIdleCPU (CPU *cpu)
{
	assert(cpu != nullptr);
	assert(cpu->_statusLock.isLockedByThisThread());
	
	assert(cpu->_activationStatus != CPU::disabled_status);
	std::lock_guard<SpinLock> guard(_idleCPUsLock);
	_idleCPUs.push_back(cpu);
}

inline void ThreadManager::unlinkIdleCPU (CPU *cpu)
{
	assert(cpu != nullptr);
	assert(cpu->_statusLock.isLockedByThisThread());
	
	std::lock_guard<SpinLock> guard(_idleCPUsLock);
	std::remove(_idleCPUs.begin(), _idleCPUs.end(), cpu);
}

inline CPU *ThreadManager::getIdleCPU()
{
	CPU *idleCPU = nullptr;
	
	std::lock_guard<SpinLock> guard(_idleCPUsLock);
	if (!_idleCPUs.empty()) {
		idleCPU = _idleCPUs.front();
		_idleCPUs.pop_front();
	}
	
	return idleCPU;
}




inline void ThreadManager::switchThreads(WorkerThread *currentThread, WorkerThread *replacementThread, CPU *cpu)
{
	assert(cpu != nullptr);
	assert(currentThread != nullptr);
	assert(currentThread->_cpu == cpu);
	assert(currentThread == WorkerThread::getCurrentWorkerThread());
	assert(cpu->_runningThread == currentThread);
	assert(cpu->_statusLock.isLockedByThisThread());
	
	if (replacementThread != nullptr) {
		if (currentThread->willResumeImmediately()) {
			currentThread->abortResumption();
			cpu->_readyThreads.push_back(currentThread);
		}
		
		assert(replacementThread->_cpu == cpu);
		replacementThread->resume();
		
		cpu->_statusLock.unlock();
		// NOTE: cpu->_runningThread may be incorrect until the replacement thread has actually set if correctly, but it will not be nullptr during that period
		
	} else {
		// No replacement thread
		if (currentThread->willResumeImmediately()) {
			currentThread->abortResumption();
			return;
		} else {
			cpu->_runningThread = nullptr;
			linkIdleCPU(cpu);
			
			cpu->_statusLock.unlock();
		}
	}
	
	currentThread->suspend();
	// After resuming, the thread continues here
	
	cpu->_statusLock.lock();
	
	// Get the CPU again, since the thread may have migrated while blocked
	CPU *cpuAfter = currentThread->_cpu;
	assert(cpuAfter != nullptr);
	
	cpuAfter->_runningThread = currentThread;
}




inline void ThreadManager::yieldIdler(WorkerThread *currentThread)
{
	assert(currentThread != nullptr);
	assert(currentThread == WorkerThread::getCurrentWorkerThread());
	
	CPU *cpu = currentThread->_cpu;
	assert(cpu != nullptr);
	
	cpu->_statusLock.lock();
	
	assert(cpu->_runningThread == currentThread);
	
	// Look up a ready thread (previosly blocked)
	WorkerThread *readyThread = getReadyThread(cpu);
	
	// Return the current thread to the idle list
	cpu->_idleThreads.push_front(currentThread);
	
	// Suspend it and replace it by the ready thread (if any)
	switchThreads(currentThread, readyThread, cpu);
	
	cpu->_statusLock.unlock();
}


inline void ThreadManager::suspendForBlocking(WorkerThread *currentThread)
{
	assert(currentThread != nullptr);
	assert(currentThread == WorkerThread::getCurrentWorkerThread());
	
	CPU *cpu = currentThread->_cpu;
	assert(cpu != nullptr);
	
	cpu->_statusLock.lock();
	
	assert(cpu->_runningThread == currentThread);
	
	// Look up a ready thread (previosly blocked)
	WorkerThread *replacementThread = getReadyThread(cpu);
	
	// If not successful, then try an idle thread. That is, assume that there may be other tasks that could be started.
	if (replacementThread == nullptr) {
		replacementThread = getNewOrIdleThread(cpu);
	}
	
	assert(replacementThread != nullptr);
	
	// Suspend the current thread and replace it by ready or idle thread
	switchThreads(currentThread, replacementThread, cpu);
	
	cpu->_statusLock.unlock();
}


inline void ThreadManager::resumeIdle(CPU *idleCPU, bool inInitialization)
{
	assert(idleCPU != nullptr);
	
	assert(idleCPU->_runningThread == nullptr);
	
	if (!inInitialization) {
		assert(WorkerThread::getCurrentWorkerThread() != nullptr);
		assert(WorkerThread::getCurrentWorkerThread()->_cpu != nullptr);
		assert(WorkerThread::getCurrentWorkerThread()->_cpu != idleCPU);
	}
	
	// Get an idle thread for the CPU
	idleCPU->_statusLock.lock();
	WorkerThread *idleThread = getNewOrIdleThread(idleCPU);
	idleCPU->_statusLock.unlock();
	
	assert(idleThread != nullptr);
	assert(idleThread->_cpu == idleCPU);
	if (!inInitialization) {
		assert(idleThread != WorkerThread::getCurrentWorkerThread());
	}
	
	// Resume it
	idleThread->resume();
}


inline void ThreadManager::resumeAnyIdle(__attribute__((unused)) HardwarePlace *preferredHardwarePlace)
{
	// FIXME: for now we are ignoring the preferredHardwarePlace
	
	bool finished = false;
	
	// This loop is necessary to remove disabled CPUs from the list of idle CPUs
	while (!finished) {
		CPU *idleCPU = getIdleCPU();
		if (idleCPU == nullptr) {
			// No idle CPUs found
			return;
		}
		
		assert(idleCPU->_activationStatus != CPU::disabled_status);
		resumeIdle(idleCPU);
		finished = true;
	}
}


inline void ThreadManager::threadBecomesReady(WorkerThread *readyThread)
{
	assert(readyThread != nullptr);
	CPU *cpu = readyThread->_cpu;
	assert(cpu != nullptr);
	
	// Assertions to avoid later resuming a thread in the current CPU (because the state was wrong)
	assert(WorkerThread::getCurrentWorkerThread() != nullptr);
	assert(WorkerThread::getCurrentWorkerThread()->_cpu != nullptr);
	assert(WorkerThread::getCurrentWorkerThread()->_cpu->_runningThread != nullptr);
	assert(WorkerThread::getCurrentWorkerThread()->_cpu->_runningThread == WorkerThread::getCurrentWorkerThread());
	
	cpu->_statusLock.lock();
	
	bool mustResume = false;
	if (cpu->_runningThread == nullptr) {
		std::lock_guard<SpinLock> guard(_idleCPUsLock);
		std::deque<CPU *>::iterator position = std::find(_idleCPUs.begin(), _idleCPUs.end(), cpu);
		if (position != _idleCPUs.end()) {
			mustResume = true;
			_idleCPUs.erase(position);
		} else {
			// A thread is being woken up
		}
	}
	
	if (mustResume) {
		readyThread->resume();
	} else {
		cpu->_readyThreads.push_back(readyThread);
	}
	
	cpu->_statusLock.unlock();
}


inline bool ThreadManager::mustExit()
{
	return _mustExit;
}


#endif // THREAD_MANAGER_HPP
