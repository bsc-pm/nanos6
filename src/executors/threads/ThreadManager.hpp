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

#include "hardware/places/ComputePlace.hpp"
#include "lowlevel/SpinLock.hpp"

#include "CPU.hpp"
#include "WorkerThread.hpp"

#include <InstrumentComputePlaceManagement.hpp>
#include <InstrumentThreadManagement.hpp>



class ThreadManagerDebuggingInterface;


class ThreadManager {
public:
	typedef std::vector<std::atomic<CPU *>> cpu_list_t;
	
private:
	//! \brief indicates if the runtime is shutting down
	static std::atomic<bool> _mustExit;
	
	//! \brief CPU mask of the process
	static cpu_set_t _processCPUMask;
	
	//! \brief per-CPU data indexed by system CPU identifier
	static cpu_list_t _cpus;
	
	//! \brief numer of initialized CPUs
	static std::atomic<long> _totalCPUs;
	
	//! \brief indicates if the thread manager has finished initializing the CPUs
	static std::atomic<bool> _finishedCPUInitialization;

	static SpinLock _idleThreadsLock;
	
	//! \brief threads blocked due to idleness
	static std::deque<WorkerThread *> _idleThreads;
	
	//! \brief number of threads in the system
	static std::atomic<long> _totalThreads;
	
	//! \brief number of threads that must be shut down
	static std::atomic<long> _shutdownThreads;
	
	//! \brief number of threads in the system that are coordinating the shutdown process
	static std::atomic<WorkerThread *> _mainShutdownControllerThread;
	
	
public:
	static void preinitialize();
	
	static void initialize();
	
	static void shutdown();
	
	
	//! \brief get or create the CPU object assigned to a given numerical system CPU identifier
	static inline CPU *getCPU(size_t systemCPUId);
	
	//! \brief get the maximum number of CPUs that will be used
	static inline long getTotalCPUs();
	
	//! \brief check if initialization has finished
	static inline bool hasFinishedInitialization();
	
	//! \brief get a reference to the CPU mask of the process
	static inline cpu_set_t const &getProcessCPUMaskReference();
	
	//! \brief get a reference to the list of CPUs
	static inline cpu_list_t const &getCPUListReference();
	
	//! \brief create or recycle a WorkerThread
	//! The thread is returned in a blocked (or about to block) status
	//!
	//! \param[in,out] cpu the CPU on which to get a new or idle thread (advisory only)
	//! \param[in] doNotCreate true to avoid creating additional threads in case that none is available
	//!
	//! \returns a WorkerThread or nullptr
	static inline WorkerThread *getIdleThread(CPU *cpu, bool doNotCreate=false);
	
	//! \brief add a thread to the list of idle threads
	//!
	//! \param[in] idleThread a thread that has become idle
	static inline void addIdler(WorkerThread *idleThread);
	
	//! \brief resume an idle thread on a given CPU
	//!
	//! \param[in] idleCPU the CPU on which to resume an idle thread
	//! \param[in] inInitializationOrShutdown true if it should not enforce assertions that are not valid during initialization and shutdown
	//! \param[in] doNotCreate true to avoid creating additional threads in case that none is available
	//!
	//! \returns the thread that has been resumed or nullptr
	static inline WorkerThread *resumeIdle(CPU *idleCPU, bool inInitializationOrShutdown=false, bool doNotCreate=false);
	
	//! \brief returns true if the thread must shut down
	static inline bool mustExit();

	//! \brief initialize a thread to run on the given CPU
	static void initializeThread(CPU *cpu);
	
	//! \brief exit the currently running thread and wake up the next one assigned to the same CPU (so that it can do the same)
	//!
	//! \param[in] currentThread a thread that is currently running and that must exit
	static void threadShutdownSequence(WorkerThread *currentThread);
	
	friend class ThreadManagerDebuggingInterface;
};

inline CPU *ThreadManager::getCPU(size_t systemCPUId)
{
	assert(systemCPUId < _cpus.size());
	
	CPU *cpu = _cpus[systemCPUId];
	assert(cpu != nullptr);
	
	return cpu;
}


inline long ThreadManager::getTotalCPUs()
{
	return _totalCPUs;
}

inline bool ThreadManager::hasFinishedInitialization()
{
	return _finishedCPUInitialization;
}


inline cpu_set_t const &ThreadManager::getProcessCPUMaskReference()
{
	return _processCPUMask;
}

inline ThreadManager::cpu_list_t const &ThreadManager::getCPUListReference()
{
	return _cpus;
}

inline WorkerThread *ThreadManager::getIdleThread(CPU *cpu, bool doNotCreate)
{
	assert(cpu != nullptr);
	
	// Try to recycle an idle thread
	{
		std::lock_guard<SpinLock> guard(_idleThreadsLock);
		if (!_idleThreads.empty()) {
			WorkerThread *idleThread = _idleThreads.front();
			_idleThreads.pop_front();
			
			assert(idleThread != nullptr);
			assert(idleThread->getTask() == nullptr);
			
			return idleThread;
		}
	}
	
	if (doNotCreate) {
		return nullptr;
	}
	
	// The shutdown code is not ready to have _totalThreads changing
	assert(!_mustExit);
	assert(_shutdownThreads == 0);
	
	// Otherwise create a new one
	_totalThreads++;
	
	// The shutdown code is not ready to have _totalThreads changing
	assert(!_mustExit);
	assert(_shutdownThreads == 0);
	
	return new WorkerThread(cpu);
}


inline void ThreadManager::addIdler(WorkerThread *idleThread)
{
	assert(idleThread != nullptr);
	
	// Return the current thread to the idle list
	{
		std::lock_guard<SpinLock> guard(_idleThreadsLock);
		_idleThreads.push_front(idleThread);
	}
}


inline WorkerThread *ThreadManager::resumeIdle(CPU *idleCPU, bool inInitializationOrShutdown, bool doNotCreate)
{
	assert(idleCPU != nullptr);
	
	if (!inInitializationOrShutdown) {
		assert((WorkerThread::getCurrentWorkerThread() == nullptr) || (WorkerThread::getCurrentWorkerThread()->_cpu != nullptr));
	}
	
	// Get an idle thread for the CPU
	WorkerThread *idleThread = getIdleThread(idleCPU, doNotCreate);
	
	if (idleThread == nullptr) {
		return nullptr;
	}
	
	idleThread->resume(idleCPU, inInitializationOrShutdown);
	
	return idleThread;
}


inline bool ThreadManager::mustExit()
{
	return _mustExit;
}


#endif // THREAD_MANAGER_HPP
