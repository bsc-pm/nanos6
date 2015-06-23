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


class ThreadManager {
	typedef threaded_executor_internals::CPU CPU;
	
	//! \brief CPU mask of the process
	static cpu_set_t _processCPUMask;
	
	//! \brief per-CPU data indexed by system CPU identifier
	static std::vector<std::atomic<CPU *>> _cpus;
	
	static SpinLock _idleCPUsLock;
	
	//! \brief CPUs that are currently idle
	static std::deque<CPU *> _idleCPUs;
	
	//! \brief retrieve an idle WorkerThread that is pinned to a given CPU
	//!
	//! \param inout cpu the CPU on which to get an idle thread
	//!
	//! \returns an idle WorkerThread pinned to the requested CPU or nullptr
	static inline WorkerThread *getIdleThread(CPU *cpu);
	
	//! \brief create or recycle a WorkerThread that is pinned to a given CPU
	//!
	//! \param inout cpu the CPU on which to get a new or idle thread
	//!
	//! \returns a WorkerThread pinned to the requested CPU
	static inline WorkerThread *getNewOrIdleThread(CPU *cpu);
	
	//! \brief get a ready WorkerThread that is pinned to a given CPU
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
	
	//! \brief resumes a thread on a CPU taking care of updating the CPU status or sets the CPU to idle (if a null thread)
	//! NOTE: This method should be called with the CPU status lock held
	//!
	//! \param in thread the thread to be resumed or nullptr to leave the CPU idle
	//! \param in cpu the affected CPU 
	static inline void resumeThread(WorkerThread *thread, CPU *cpu);
	
	static inline void linkIdleCPU(CPU *cpu);
	static inline void unlinkIdleCPU(CPU *cpu);
	static inline CPU *getIdleCPU();
	
	
public:
	static void initialize();
	
	static void shutdown();
	
	
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
	
	//! \brief attempt to activate an idle CPU
	//!
	//! \param in preferredHardwarePlace a hint on a desirable area of the hardware
	static inline void resumeAnyIdle(HardwarePlace *preferredHardwarePlace);
	
	//! \brief indicate that a previously ready thread has become ready
	//!
	//! \param in readyThread a thready that was blocked and now is ready to be executed again
	static inline void threadBecomesReady(WorkerThread *readyThread);
	
	
	//! \brief set a CPU online
	static void enableCPU(size_t systemCPUId);
	
	//! \brief set a CPU offline
	static void disableCPU(size_t systemCPUId);
	
	
	//! \brief returns true if the thread must shut down
	//!
	//! \param in cpu the CPU that is running the current thread
	static inline bool mustExit(CPU *cpu);
	
	//! \brief exit the currently running thread and wake up the next one assigned to the same CPU (so that it can do the same)
	//!
	//! \param in currentThread a thread that is currently running and that must exit
	static void exitAndWakeUpNext(WorkerThread *currentThread);
};


//
// Getting / Returning worker threads
//


inline WorkerThread *ThreadManager::getIdleThread(CPU *cpu)
{
	assert(cpu != nullptr);
	
	std::lock_guard<SpinLock> guard(cpu->_idleThreadsLock);
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
	
	WorkerThread *idleThread = getIdleThread(cpu);
	if (idleThread != nullptr) {
		return idleThread;
	}
	
	WorkerThread *newThread = new WorkerThread(cpu);
	
	return newThread;
}

inline WorkerThread *ThreadManager::getReadyThread(CPU *cpu)
{
	assert(cpu != nullptr);
	WorkerThread *readyThread = nullptr;
	
	{
		std::lock_guard<SpinLock> guard(cpu->_readyThreadsLock);
		if (!cpu->_readyThreads.empty()) {
			readyThread = cpu->_readyThreads.front();
			cpu->_readyThreads.pop_front();
		}
	}
	
	return readyThread;
}


inline void ThreadManager::resumeThread(WorkerThread *thread, CPU *cpu)
{
	assert(cpu != nullptr);
	cpu->_runningThread = thread; // NOTE: this will set it to null if (thread == nullptr);
	if (thread != nullptr) {
		assert(thread->_cpu == cpu); // NOTE: until we start migrating threads
		thread->resume();
	} else {
		linkIdleCPU(cpu);
	}
}


inline void ThreadManager::linkIdleCPU (CPU *cpu)
{
	assert(cpu != nullptr);
	
	std::lock_guard<SpinLock> guard(_idleCPUsLock);
	_idleCPUs.push_back(cpu);
}

inline void ThreadManager::unlinkIdleCPU (ThreadManager::CPU *cpu)
{
	assert(cpu != nullptr);
	
	std::lock_guard<SpinLock> guard(_idleCPUsLock);
	std::remove(_idleCPUs.begin(), _idleCPUs.end(), cpu);
}

inline ThreadManager::CPU *ThreadManager::getIdleCPU()
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
	
	// Resume the replacement thread (if any)
	{
		std::lock_guard<SpinLock> guard(cpu->_statusLock);
		resumeThread(replacementThread, cpu); // Also updates the status in case that replacementThread is nullptr
	}
	
	// Suspend the thread
	bool presignaled = currentThread->suspend();
	
	// After resuming, the thread continues here
	
	// If it had been presignaled but we are waking up another, place it in the ready list and suspend it again
	if (presignaled && (replacementThread != nullptr)) {
		threadBecomesReady(currentThread);
		currentThread->suspend();
		// If presignaled a second time, either there is a bug, or
		// the thread was woken up after adding it to the ready queue but before suspending it
	}
	
	// Get the CPU again, since the thread may have migrated while blocked
	CPU *cpuAfter = currentThread->_cpu;
	assert(cpuAfter != nullptr);
	
	if (presignaled && (replacementThread == nullptr)) {
		// Fix the cpu to thread assignment
		cpuAfter->_runningThread = currentThread;
	}
	
	// NOTE: Right before resuming a thread we always set it up as the thread running on the CPU
	assert(cpuAfter->_runningThread == currentThread);
}




inline void ThreadManager::yieldIdler(WorkerThread *currentThread)
{
	assert(currentThread != nullptr);
	assert(currentThread == WorkerThread::getCurrentWorkerThread());
	
	CPU *cpu = currentThread->_cpu;
	
	assert(cpu != nullptr);
	assert(cpu->_runningThread == currentThread);
	
	// Look up a ready thread (previosly blocked)
	WorkerThread *readyThread = getReadyThread(cpu);
	
	// Return the current thread to the idle list
	{
		std::lock_guard<SpinLock> guard(cpu->_idleThreadsLock);
		cpu->_idleThreads.push_front(currentThread);
	}
	
	// Suspend it and replace it by the ready thread (if any)
	switchThreads(currentThread, readyThread, cpu);
}


inline void ThreadManager::suspendForBlocking(WorkerThread *currentThread)
{
	assert(currentThread != nullptr);
	assert(currentThread == WorkerThread::getCurrentWorkerThread());
	
	CPU *cpu = currentThread->_cpu;
	assert(cpu != nullptr);
	assert(cpu->_runningThread == currentThread);
	
	// FIXME: Handle !cpu->_enabled
	
	// Look up a ready thread (previosly blocked)
	WorkerThread *replacementThread = getReadyThread(cpu);
	
	// If not successful, then try an idle thread. That is, assume that there may be other tasks that could be started.
	if (replacementThread == nullptr) {
		replacementThread = getNewOrIdleThread(cpu);
	}
	
	assert(replacementThread != nullptr);
	
	// Suspend the current thread and replace it by ready or idle thread
	switchThreads(currentThread, replacementThread, cpu);
}


inline void ThreadManager::resumeAnyIdle(__attribute__((unused)) HardwarePlace *preferredHardwarePlace)
{
	// FIXME: for now we are ignoring the preferredHardwarePlace
	
	CPU *idleCPU = getIdleCPU();
	if (idleCPU == nullptr) {
		// No idle CPUs found
		return;
	}
	
	// Get an idle thread for the CPU
	WorkerThread *idleThread = getNewOrIdleThread(idleCPU);
	assert(idleThread != nullptr);
	assert(idleThread->_cpu == idleCPU);
	
	// Resume it
	std::lock_guard<SpinLock> guard(idleCPU->_statusLock);
	resumeThread(idleThread, idleCPU);
}


inline void ThreadManager::threadBecomesReady(WorkerThread *readyThread)
{
	CPU *cpu = readyThread->_cpu;
	assert(cpu != nullptr);
	
	cpu->_statusLock.lock();
	if (cpu->_runningThread == nullptr) {
		resumeThread(readyThread, cpu);
		cpu->_statusLock.unlock();
	} else {
		cpu->_statusLock.unlock();
		std::lock_guard<SpinLock> guard(cpu->_readyThreadsLock);
		cpu->_readyThreads.push_back(readyThread);
	}
}


inline bool ThreadManager::mustExit(CPU *currentCPU)
{
	assert(currentCPU != nullptr);
	
	std::lock_guard<SpinLock> guard(currentCPU->_statusLock); // This is necessary to avoid a race condition during shutdown. Since threads are pinned, the same exact CPU is always the one that accesses the given SpinLock
	return currentCPU->_mustExit;
}


#endif // THREAD_MANAGER_HPP
