#include <cassert>

#include <pthread.h>
#include <unistd.h>

#include "ThreadManager.hpp"


cpu_set_t ThreadManager::_processCPUMask;
std::vector<std::atomic<ThreadManager::CPU *>> ThreadManager::_cpus(CPU_SETSIZE);
SpinLock ThreadManager::_idleCPUsLock;
std::deque<ThreadManager::CPU *> ThreadManager::_idleCPUs;


void ThreadManager::initialize()
{
	int rc = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &_processCPUMask);
	assert(rc == 0);
	
	// Set up the pthread attributes for the threads of each CPU
	for (size_t systemCPUId = 0; systemCPUId < CPU_SETSIZE; systemCPUId++) {
		if (CPU_ISSET(systemCPUId, &_processCPUMask)) {
			enableCPU(systemCPUId);
		}
	}
}


void ThreadManager::shutdown()
{
	for (CPU *cpu: _cpus) {
		if (cpu != nullptr) {
			{
				std::lock_guard<SpinLock> guard(cpu->_readyThreadsLock);
				assert(cpu->_readyThreads.empty());
			}
			
			std::lock_guard<SpinLock> guard(cpu->_statusLock);
			
			cpu->_mustExit = true;
			
			if (cpu->_enabled) {
				if (cpu->_runningThread == nullptr) {
					WorkerThread *thread = getIdleThread(cpu);
					resumeThread(thread, cpu);
				} else {
					// In principle there is a thread running that will either detect the shutdown signal or end up waking up another thread that will notice it
				}
			} else {
				// Disabled CPU
				std::lock_guard<SpinLock> guard2(cpu->_idleThreadsLock);
				for (auto idleThread: cpu->_idleThreads) {
					int rc = pthread_cancel(idleThread->_pthread);
					assert(rc == 0);
				}
			}
		}
	}
}


//
// Setting CPUs online / offline
//

void ThreadManager::enableCPU(size_t systemCPUId)
{
	if (_cpus[systemCPUId] == nullptr) {
		CPU *currentCPUObject = nullptr;
		
		CPU *newCPU = new CPU(systemCPUId);
		_cpus[systemCPUId].compare_exchange_strong(currentCPUObject, newCPU);
		
		assert(currentCPUObject == nullptr); // Another thread enabled the CPU !?!
	} else {
		assert(!((CPU *)_cpus[systemCPUId])->_enabled);
	}
	
	CPU *cpu = _cpus[systemCPUId];
	{
		std::lock_guard<SpinLock> guard(cpu->_statusLock);
		assert(!cpu->_enabled);
		cpu->_enabled = true;
		
		WorkerThread *thread = getNewOrIdleThread(cpu);
		thread->resume();
	}
}

void ThreadManager::disableCPU(size_t systemCPUId)
{
	CPU *cpu = _cpus[systemCPUId];
	assert(cpu != nullptr);
	
	std::lock_guard<SpinLock> guard(cpu->_statusLock);
	assert(cpu->_enabled);
	cpu->_enabled = false;
}


void ThreadManager::exitAndWakeUpNext (WorkerThread *currentThread)
{
	CPU *cpu = currentThread->_cpu;
	assert(cpu != nullptr);
	assert(cpu->_runningThread == currentThread);
	assert(WorkerThread::getCurrentWorkerThread() == currentThread);
	
	// Find next to wake
	WorkerThread *next = getIdleThread(cpu);
	
	// Resume it, or set the CPU to idle
	{
		std::lock_guard<SpinLock> guard(cpu->_statusLock);
		resumeThread(next, cpu);
	}
	
	// Exit the current thread
	currentThread->exit();
}

