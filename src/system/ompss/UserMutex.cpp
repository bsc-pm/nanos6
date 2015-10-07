#include "api/nanos6_rt_interface.h"

#include "TaskBlocking.hpp"
#include "UserMutex.hpp"

#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/ThreadManagerPolicy.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "lowlevel/SpinLock.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/Task.hpp"

#include <InstrumentUserMutex.hpp>

#include <cassert>


typedef std::atomic<UserMutex *> mutex_t;


void nanos_user_lock(void **handlerPointer, __attribute__((unused)) char const *invocationSource)
{
	assert(handlerPointer != nullptr);
	mutex_t &userMutexReference = (mutex_t &) *handlerPointer;
	
	// Allocation
	if (__builtin_expect(userMutexReference == nullptr, 0)) {
		UserMutex *newMutex = new UserMutex(true);
		
		UserMutex *expected = nullptr;
		if (userMutexReference.compare_exchange_strong(expected, newMutex)) {
			// Successfully assigned new mutex
			assert(userMutexReference == newMutex);
			
			Instrument::acquiredUserMutex(newMutex);
			
			// Since we allocate the mutex in the locked state, the thread already owns it and the work is done
			return;
		} else {
			// Another thread managed to initialize it before us
			assert(expected != nullptr);
			assert(userMutexReference == expected);
			
			delete newMutex;
			
			// Continue through the "normal" path
		}
	}
	
	// The mutex has already been allocated and cannot change, so skip the atomic part from now on
	UserMutex &userMutex = *(userMutexReference.load());
	
	// Fast path
	if (userMutex.tryLock()) {
		Instrument::acquiredUserMutex(&userMutex);
		return;
	}
	
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);
	
	Task *currentTask = currentThread->getTask();
	assert(currentTask != nullptr);
	
	// Acquire the lock if possible. Otherwise queue the task.
	if (userMutex.lockOrQueue(currentTask)) {
		// Successful
		Instrument::acquiredUserMutex(&userMutex);
		return;
	}
	
	Instrument::blockedOnUserMutex(&userMutex);
	TaskBlocking::taskBlocks(currentThread, currentTask);
	Instrument::acquiredUserMutex(&userMutex);
}


void nanos_user_unlock(void **handlerPointer)
{
	assert(handlerPointer != nullptr);
	assert(*handlerPointer != nullptr);
	
	mutex_t &userMutexReference = (mutex_t &) *handlerPointer;
	UserMutex &userMutex = *(userMutexReference.load());
	Instrument::releasedUserMutex(&userMutex);
	
	Task *releasedTask = userMutex.dequeueOrUnlock();
	if (releasedTask != nullptr) {
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		assert(currentThread != nullptr);
		
		CPU *cpu = currentThread->getHardwarePlace();
		assert(cpu != nullptr);
		
		Task *currentTask = currentThread->getTask();
		assert(currentTask != nullptr);
		
		if (ThreadManagerPolicy::checkIfUnblockedMustPreemtUnblocker(currentTask, releasedTask, cpu)) {
			WorkerThread *releasedThread = releasedTask->getThread();
			assert(releasedThread != nullptr);
			
			CPU *idleCPU = (CPU *) Scheduler::getIdleHardwarePlace();
			if (idleCPU != nullptr) {
				// Wake up the unblocked task and migrate to an idle CPU
				ThreadManager::resumeThread(releasedThread, cpu);
				ThreadManager::migrateThread(currentThread, idleCPU);
			} else {
				Scheduler::taskGetsUnblocked(currentTask, cpu);
				
				ThreadManager::switchThreads(currentThread, releasedThread);
			}
		} else {
			Scheduler::taskGetsUnblocked(releasedTask, cpu);
			
			CPU *idleCPU = (CPU *) Scheduler::getIdleHardwarePlace();
			if (idleCPU != nullptr) {
				ThreadManager::resumeIdle(idleCPU);
			}
		}
		
		// WARNING: cpu is no longer valid here, refresh its value if needed
	}
}
