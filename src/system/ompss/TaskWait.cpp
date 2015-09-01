#include "api/nanos6_rt_interface.h"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/ThreadManagerPolicy.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/Task.hpp"

#include <cassert>



void nanos_taskwait()
{
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);
	
	Task *currentTask = currentThread->getTask();
	assert(currentTask != nullptr);
	assert(currentTask->getThread() == currentThread);
	
	CPU *cpu = currentThread->getHardwarePlace();
	assert(cpu != nullptr);
	
	// Fast check
	if (currentTask->doesNotNeedToBlockForChildren()) {
		return;
	}
	
	bool done = currentTask->markAsBlocked();
	
	// done == true:
	// 	1. The condition of the taskwait has been fulfilled
	// 	2. The task will not be queued at all
	// 	3. The execution must continue (without blocking)
	// done == false:
	// 	1. The task has been marked as blocked
	// 	2. At any time the condition of the taskwait can become true
	// 	3. The thread responsible for that change will queue the task
	// 	4. Any thread can dequeue it and attempt to resume the thread
	// 	5. This can trigger a migration, and will make the call to ThreadManager::switchThreads to resume immediately (and to wake the replacement thread, if any, on the "old" CPU)
	
	while (!done) {
		Task *replacementTask = nullptr;
		
		// The task needs to block. However during the following code, it can reach an unblocking condition.
		// This can cause another thread to migrate it to another CPU and to presignal it, which will cause
		// it to not block in the call to switchThreads.
		
		replacementTask = Scheduler::getReadyTask(cpu, currentTask);
		
		// Case 0: The current task just got woken up and by chance
		// has been returned to its own thread. So just keep running
		if (replacementTask == currentTask) {
			done = true;
			break;
		}
		
		// The following variable will end up containing either:
		// (1) a thread with a(n already started) replacement task assigned
		// (2) an idle thread with the task pre-assigned
		// (3) nullptr if there is a replacement task and it must be run within this thread
		WorkerThread *replacementThread = nullptr;
		
		bool runReplacementInline = false;
		if (replacementTask != nullptr) {
			// Attempt to set up case (1)
			replacementThread = replacementTask->getThread();
			
			if (replacementThread == nullptr) {
				runReplacementInline = ThreadManagerPolicy::checkIfMustRunInline(replacementTask, currentTask, cpu);
				if (!runReplacementInline) {
					// Set up case (2)
					replacementThread = ThreadManager::getIdleThread(cpu);
					replacementThread->setTask(replacementTask);
				}
			}
		}
		
		if (runReplacementInline) {
			assert(replacementThread == nullptr);
			
			currentThread->handleTask(replacementTask);
			
			// The thread can have migrated while running the replacement task
			cpu = currentThread->getHardwarePlace();
			
			if (currentTask->canBeWokenUp()) {
				// The task has or is entering the unblocked queue.
				// So switch to an idle thread that will remove it from there and wake it up.
				//
				// NOTE: This could be optimized by having a call in the scheduler to do it
				// and cleaning up the immediate resumption flag. However we do not have any
				// guarantee of how long it is going to take to have the task in the queue,
				// and the thread (pre-)signaled.
				replacementThread = ThreadManager::getIdleThread(cpu);
				ThreadManager::switchThreads(currentThread, replacementThread);
				
				// At this point the condition of the taskwait has been fulfilled
				done = true;
			}
		} else {
			// Either switch to the replacement thread, or just get blocked
			ThreadManager::switchThreads(currentThread, replacementThread);
			
			// At this point the condition of the taskwait has been fulfilled
			done = true;
		}
	}
	
	assert(currentTask->canBeWokenUp());
	currentTask->markAsUnblocked();
}

