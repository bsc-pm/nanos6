#include "ThreadManager.hpp"
#include "WorkerThread.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/Task.hpp"


#include <pthread.h>


__thread WorkerThread *WorkerThread::_currentWorkerThread = nullptr;


WorkerThread::WorkerThread(WorkerThread::CPU *cpu)
	: _suspensionConditionVariable(), _cpu(cpu), _task(nullptr)
{
	int rc = pthread_create(&_pthread, &_cpu->_pthreadAttr, (void* (*)(void*)) &WorkerThread::body, this);
	assert(rc == 0);
}


void *WorkerThread::body()
{
	_currentWorkerThread = this;
	
	suspend();
	
	_cpu->_runningThread = this;
	
	while (!ThreadManager::mustExit(_cpu)) {
		_task = Scheduler::schedule(_cpu);
		
		if (_task != nullptr) {
			handleTask();
		} else {
			ThreadManager::yieldIdler(this);
		}
	}
	
	ThreadManager::exitAndWakeUpNext(this);
	
	assert(false);
	return nullptr;
}


void WorkerThread::handleTask()
{
	// Run the task
	_task->setThread(this);
	_task->body();
	
	// TODO: Release successors
	
	// Follow up the chain of ancestors and dispose them as needed and wake up any in a taskwait that finishes in this moment
	{
		bool canDispose = _task->markAsFinished();
		Task *potentiallyDisposableTask = _task;
		
		if (canDispose) {
			while (potentiallyDisposableTask != nullptr) {
				Task *parent = potentiallyDisposableTask->getParent();
				
				bool parentIsReadyOrDisposable = potentiallyDisposableTask->unlinkFromParent();
				if (parentIsReadyOrDisposable) {
					if (potentiallyDisposableTask->hasFinished()) {
						delete potentiallyDisposableTask; // FIXME: Need a proper object recycling mechanism here
						potentiallyDisposableTask = parent;
					} else {
						// An ancestor in a taskwait that finishes at this point
						ThreadManager::threadBecomesReady(potentiallyDisposableTask->getThread());
						potentiallyDisposableTask = nullptr;
					}
				} else {
					potentiallyDisposableTask = nullptr;
				}
			}
		}
	}
	
	_task = nullptr;
}

