#include "api/nanos6_rt_interface.h"

#include "TaskBlocking.hpp"

#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/ThreadManagerPolicy.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "lowlevel/SpinLock.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/Task.hpp"

#include <atomic>
#include <cassert>
#include <deque>


namespace nanos6 {
	class UserMutex {
		//! \brief The user mutex state
		std::atomic<bool> _userMutex;
		
		//! \brief The spin lock that protects the queue of tasks blocked on this user-side mutex
		SpinLock _blockedTasksLock;
		
		//! \brief The list of tasks blocked on this user-side mutex
		std::deque<Task *> _blockedTasks;
		
	public:
		//! \brief Initialize the mutex
		//!
		//! \param[in] initialState true if the mutex must be initialized in the locked state
		inline UserMutex(bool initialState)
			: _userMutex(initialState), _blockedTasksLock(), _blockedTasks()
		{
		}
		
		//! \brief Try to lock
		//!
		//! \returns true if the user-lock has been locked successfully, false otherwise
		inline bool tryLock()
		{
			bool expected = false;
			bool successful = _userMutex.compare_exchange_strong(expected, false);
			assert(expected != successful);
			return successful;
		}
		
		//! \brief Try to lock of queue the task
		//!
		//! \param[in] task The task that will be queued if the lock cannot be acquired
		//!
		//! \returns true if the lock has been acquired, false if not and the task has been queued
		inline bool lockOrQueue(Task *task)
		{
			std::lock_guard<SpinLock> guard(_blockedTasksLock);
			if (tryLock()) {
				return true;
			} else {
				_blockedTasks.push_back(task);
				return false;
			}
		}
		
		inline Task *dequeueOrUnlock()
		{
			std::lock_guard<SpinLock> guard(_blockedTasksLock);
			
			if (_blockedTasks.empty()) {
				_userMutex = false;
				return nullptr;
			}
			
			Task *releasedTask = _blockedTasks.front();
			_blockedTasks.pop_front();
			assert(releasedTask != nullptr);
			
			return releasedTask;
		}
	};
	
	typedef std::atomic<UserMutex *> mutex_t;
}


using namespace nanos6;


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
		return;
	}
	
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);
	
	Task *currentTask = currentThread->getTask();
	assert(currentTask != nullptr);
	
	// Acquire the lock if possible. Otherwise queue the task.
	if (userMutex.lockOrQueue(currentTask)) {
		// Successful
		return;
	}
	
	TaskBlocking::taskBlocks(currentThread, currentTask);
}


void nanos_user_unlock(void **handlerPointer)
{
	assert(handlerPointer != nullptr);
	assert(*handlerPointer != nullptr);
	
	mutex_t &userMutexReference = (mutex_t &) *handlerPointer;
	UserMutex &userMutex = *(userMutexReference.load());
	
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

