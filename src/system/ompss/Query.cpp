#include "api/nanos6_rt_interface.h"

#include "executors/threads/WorkerThread.hpp"
#include "tasks/Task.hpp"

#include <cassert>


signed int nanos_in_final(void)
{
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);
	
	Task *currentTask = currentThread->getTask();
	assert(currentTask != nullptr);
	assert(currentTask->getThread() == currentThread);
	
	return currentTask->isFinal();
}

