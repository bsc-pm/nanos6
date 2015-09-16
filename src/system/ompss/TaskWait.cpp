#include "api/nanos6_rt_interface.h"

#include "TaskBlocking.hpp"

#include "executors/threads/WorkerThread.hpp"
#include "tasks/Task.hpp"

#include <InstrumentTaskWait.hpp>

#include <cassert>



void nanos_taskwait(__attribute__((unused)) char const *invocationSource)
{
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);
	
	Task *currentTask = currentThread->getTask();
	assert(currentTask != nullptr);
	assert(currentTask->getThread() == currentThread);
	
	Instrument::enterTaskWait(currentTask->getInstrumentationTaskId(), invocationSource);
	
	// Fast check
	if (currentTask->doesNotNeedToBlockForChildren()) {
		Instrument::exitTaskWait(currentTask->getInstrumentationTaskId());
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
	// 	5. This can trigger a migration, and will make the call to
	// 		ThreadManager::switchThreads (that is inside TaskBlocking::taskBlocks)
	// 		to resume immediately (and to wake the replacement thread, if any,
	// 		on the "old" CPU)
	
	if (!done) {
		TaskBlocking::taskBlocks(currentThread, currentTask);
	}
	
	Instrument::exitTaskWait(currentTask->getInstrumentationTaskId());
	
	assert(currentTask->canBeWokenUp());
	currentTask->markAsUnblocked();
}

