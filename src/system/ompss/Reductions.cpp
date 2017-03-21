#include <nanos6.h>

#include <cassert>

#include "tasks/Task.hpp"
#include "executors/threads/WorkerThread.hpp"


void *nanos_get_original_reduction_address(const void *address)
{
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);
	
	Task *currentTask = currentThread->getTask();
	assert(currentTask != nullptr);
	assert(currentTask->getThread() == currentThread);
	
	void *argsBlock = currentTask->getArgsBlock();
	
	if (argsBlock <= address && currentTask > address) {
		return (void*)*(((void**)address) - 1);
	}
	else {
		return (void*)address;
	}
}
