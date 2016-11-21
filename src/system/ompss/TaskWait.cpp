#include "api/nanos6_rt_interface.h"

#include "DataAccessRegistration.hpp"
#include "TaskBlocking.hpp"

#include "executors/threads/WorkerThread.hpp"
#include "tasks/Task.hpp"

#include "hardware/Machine.hpp"

#include <InstrumentTaskWait.hpp>
#include <InstrumentTaskStatus.hpp>

#include <cassert>



void nanos_taskwait(__attribute__((unused)) char const *invocationSource)
{
	Task *currentTask = nullptr;
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	
	assert(currentThread != nullptr);
	
	currentTask = currentThread->getTask();
	assert(currentTask != nullptr);
	assert(currentTask->getThread() == currentThread);
	
	Instrument::enterTaskWait(currentTask->getInstrumentationTaskId(), invocationSource);
	
	// Fast check
	if (currentTask->doesNotNeedToBlockForChildren()) {
		// This in combination with a release from the children makes their changes visible to this thread
		std::atomic_thread_fence(std::memory_order_acquire);
		
		Instrument::exitTaskWait(currentTask->getInstrumentationTaskId());
		
		return;
	}
	
	DataAccessRegistration::handleEnterTaskwait(currentTask);
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
		Instrument::taskIsBlocked(currentTask->getInstrumentationTaskId(), Instrument::in_taskwait_blocking_reason);
		TaskBlocking::taskBlocks(currentThread, currentTask, true);
	}
	
	// This in combination with a release from the children makes their changes visible to this thread
	std::atomic_thread_fence(std::memory_order_acquire);
	
	Instrument::exitTaskWait(currentTask->getInstrumentationTaskId());
	
	assert(currentTask->canBeWokenUp());
	currentTask->markAsUnblocked();
	
	DataAccessRegistration::handleExitTaskwait(currentTask);
    GenericCache * destCache = currentTask->getCache();
    destCache->flush();

	
	if (!done && (currentThread != nullptr)) {
		// The instrumentation was notified that the task had been blocked
		Instrument::taskIsExecuting(currentTask->getInstrumentationTaskId());
	}
}

