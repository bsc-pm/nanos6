/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "TaskBlocking.hpp"

#include "executors/threads/CPU.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/ThreadManagerPolicy.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"

#include <cassert>


void TaskBlocking::taskBlocks(WorkerThread *currentThread, Task *currentTask, ThreadManagerPolicy::thread_run_inline_policy_t policy)
{
	assert(currentThread != nullptr);
	assert(currentTask != nullptr);
	
	assert(WorkerThread::getCurrentWorkerThread() == currentThread);
	assert(currentThread->getTask() == currentTask);
	
	CPU *cpu = currentThread->getComputePlace();
	assert(cpu != nullptr);
	
	bool done = false;
	while (!done) {
		Task *replacementTask = nullptr;
		
		// The task needs to block. However during the following code, it can reach an unblocking condition.
		// This can cause another thread to migrate it to another CPU and to presignal it, which will cause
		// it to not block in the call to switchTo.
		
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
				runReplacementInline = ThreadManagerPolicy::checkIfMustRunInline(replacementTask, currentTask, cpu, policy);
				
				if (!runReplacementInline) {
					// Set up case (2)
					replacementThread = ThreadManager::getIdleThread(cpu);
					replacementThread->setTask(replacementTask);
				}
			}
		}
		
		if (runReplacementInline) {
			assert(replacementThread == nullptr);
			
			// The blocking condition may released while this thread is running a task. Avoid that
			if (currentTask->disableScheduling()) {
				currentThread->handleTask(cpu, replacementTask);
				
				// The thread can have migrated while running the replacement task
				cpu = currentThread->getComputePlace();
				
				if (currentTask->enableScheduling()) {					
					// At this point the blocking condition has been fulfilled. The task is not in the scheduler
					done = true;
				}
			} else {
				// The task has or is entering the unblocked queue.
				// Run the task that was obtained in a new thread, so this thread can be woken up
				replacementThread = ThreadManager::getIdleThread(cpu);
				replacementThread->setTask(replacementTask);
				currentThread->switchTo(replacementThread);
				done = true;
			}
		} else {
			// Either switch to the replacement thread, or just get blocked
			currentThread->switchTo(replacementThread);
			
			// At this point the condition of the taskwait has been fulfilled
			done = true;
		}
	}
	
}
