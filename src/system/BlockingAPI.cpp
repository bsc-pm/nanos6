#include <cassert>

#include <nanos6/blocking.h>

#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"

#include "scheduling/Scheduler.hpp"


extern "C" void *nanos_get_current_task()
{
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	assert(currentWorkerThread != nullptr);
	
	Task *task = currentWorkerThread->getTask();
	assert(task != nullptr);
	
	return task;
}


extern "C" void nanos_block_current_task()
{
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	assert(currentWorkerThread != nullptr);
	
	CPU *cpu = nullptr;
	cpu = currentWorkerThread->getComputePlace();
	assert(cpu != nullptr);
	
	WorkerThread *replacementThread = ThreadManager::getIdleThread(cpu);
	
	ThreadManager::switchThreads(currentWorkerThread, replacementThread);
}


extern "C" void nanos_unblock_task(void *blocked_task_handler)
{
	Task *task = static_cast<Task *>(blocked_task_handler);
	
	Scheduler::taskGetsUnblocked(task, nullptr);
}

