/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6.h>

#include "executors/threads/WorkerThread.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"

#include <cassert>


signed int nanos6_in_final(void)
{
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);
	
	Task *currentTask = currentThread->getTask();
	assert(currentTask != nullptr);
	assert(currentTask->getThread() == currentThread);
	
	return currentTask->isFinal();
}

signed int nanos6_in_serial_context(void)
{
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);
	
	Task *currentTask = currentThread->getTask();
	assert(currentTask != nullptr);
	assert(currentTask->getThread() == currentThread);
	
	return currentTask->isFinal() || currentTask->isIf0();
}

