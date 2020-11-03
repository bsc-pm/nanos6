/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_BLOCKING_HPP
#define TASK_BLOCKING_HPP

#include "executors/threads/CPU.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "system/TrackingPoints.hpp"

class TaskBlocking {

public:

	static inline void taskBlocks(
		WorkerThread *currentThread,
		__attribute__((unused)) Task *currentTask
	) {
		assert(currentTask != nullptr);
		assert(currentThread != nullptr);
		assert(currentThread->getTask() == currentTask);
		assert(WorkerThread::getCurrentWorkerThread() == currentThread);

		CPU *cpu = currentThread->getComputePlace();
		assert(cpu != nullptr);

		WorkerThread *replacementThread = ThreadManager::getIdleThread(cpu);
		assert(replacementThread != nullptr);

		// Runtime Tracking Point - Thread is suspending
		TrackingPoints::threadWillSuspend(currentThread, cpu);

		// When a task blocks, switch to another idle thread to avoid:
		// 1) Getting the current thread stuck in the CPU while doing nothing
		// 2) Assigning replacement tasks to threads
		currentThread->switchTo(replacementThread);
	}

};

#endif // TASK_BLOCKING_HPP
