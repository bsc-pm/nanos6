#include <cassert>

#include <pthread.h>
#include <unistd.h>

#include "CPUActivation.hpp"
#include "ThreadManager.hpp"


std::atomic<bool> ThreadManager::_mustExit;
cpu_set_t ThreadManager::_processCPUMask;
std::vector<std::atomic<ThreadManager::CPU *>> ThreadManager::_cpus(CPU_SETSIZE);
SpinLock ThreadManager::_idleCPUsLock;
std::deque<ThreadManager::CPU *> ThreadManager::_idleCPUs;


void ThreadManager::initialize()
{
	_mustExit = false;
	
	int rc = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &_processCPUMask);
	assert(rc == 0);
	
	// Set up the pthread attributes for the threads of each CPU
	for (size_t systemCPUId = 0; systemCPUId < CPU_SETSIZE; systemCPUId++) {
		if (CPU_ISSET(systemCPUId, &_processCPUMask)) {
			CPU *cpu = getCPU(systemCPUId);
			
			assert(cpu != nullptr);
			resumeIdle(cpu, true);
		}
	}
}


void ThreadManager::shutdown()
{
	_mustExit = true;
	
	for (CPU *cpu: _cpus) {
		if (cpu != nullptr) {
			std::lock_guard<SpinLock> guard(cpu->_statusLock);
			
			assert(cpu->_readyThreads.empty());
			
			while ((cpu->_activationStatus != CPU::enabled_status) && (cpu->_activationStatus != CPU::disabled_status)) {
				// Wait for the CPU status to make the transition
			}
			
			if (cpu->_activationStatus == CPU::enabled_status) {
				if (cpu->_runningThread == nullptr) {
					WorkerThread *thread = getIdleThread(cpu);
					
					if (thread != nullptr) {
						assert(thread->_cpu == cpu);
						thread->resume();
					}
				} else {
					// In principle there is a thread running that will either detect the shutdown signal or end up waking up another thread that will notice it
				}
			} else if (cpu->_activationStatus == CPU::disabled_status) {
				// Disabled CPU
				for (auto idleThread: cpu->_idleThreads) {
					int rc = pthread_cancel(idleThread->_pthread);
					assert(rc == 0);
				}
			} else {
				assert("The CPU status was transitioning during shutdown" == nullptr);
			}
		}
	}
}


void ThreadManager::threadStartup(WorkerThread *currentThread)
{
	assert(currentThread != nullptr);
	
	CPU *cpu = currentThread->_cpu;
	
	assert(cpu != nullptr);
	
	WorkerThread::_currentWorkerThread = currentThread;
	
	// Initialize the CPU status if necessary before the thread has a chance to check the shutdown signaled
	CPUActivation::activationCheck(currentThread);
	
	// The thread suspends itself after initialization, since the "activator" is the one will unblock it when needed
	currentThread->suspend();
	
	std::lock_guard<SpinLock> guard(cpu->_statusLock);
	cpu->_runningThread = currentThread;
}


void ThreadManager::exitAndWakeUpNext(WorkerThread *currentThread)
{
	CPU *cpu = currentThread->_cpu;
	assert(cpu != nullptr);
	assert(cpu->_runningThread == currentThread);
	assert(WorkerThread::getCurrentWorkerThread() == currentThread);
	
	{
		std::lock_guard<SpinLock> guard(cpu->_statusLock);
		
		// Find next to wake
		WorkerThread *next = getIdleThread(cpu);
		
		if (next != nullptr) {
			assert(next->_cpu == cpu);
			next->resume();
		}
	}
	
	// Exit the current thread
	currentThread->exit();
}

