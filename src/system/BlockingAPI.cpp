#include <cassert>

#include <nanos6/blocking.h>

#include "DataAccessRegistration.hpp"
#include "ompss/TaskBlocking.hpp"

#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"

#include "scheduling/Scheduler.hpp"


extern "C" void *nanos_get_current_blocking_context()
{
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);
	
	Task *currentTask = currentThread->getTask();
	assert(currentTask != nullptr);
	
	return currentTask;
}


extern "C" void nanos_block_current_task(void *blocking_context)
{
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);
	
	CPU *cpu = nullptr;
	cpu = currentThread->getHardwarePlace();
	assert(cpu != nullptr);
	
	Task *currentTask = currentThread->getTask();
	assert(currentTask != nullptr);
	
	assert(blocking_context == currentTask);
	
	DataAccessRegistration::handleEnterBlocking(currentTask);
	TaskBlocking::taskBlocks(currentThread, currentTask, false);
	DataAccessRegistration::handleExitBlocking(currentTask);
}


extern "C" void nanos_unblock_task(void *blocking_context)
{
	Task *task = static_cast<Task *>(blocking_context);
	
	Scheduler::taskGetsUnblocked(task, nullptr);
	
	CPU *idleCPU = (CPU *) Scheduler::getIdleHardwarePlace();
	if (idleCPU != nullptr) {
		ThreadManager::resumeIdle(idleCPU);
	}
}

