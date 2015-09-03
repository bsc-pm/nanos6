#include "CPUActivation.hpp"
#include "ThreadManager.hpp"
#include "WorkerThread.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/Task.hpp"


#include <pthread.h>


__thread WorkerThread *WorkerThread::_currentWorkerThread = nullptr;


static void *worker_thread_body_wrapper(void *parameter)
{
	WorkerThread *workerThread = (WorkerThread *) parameter;
	assert(workerThread != nullptr);
	return workerThread->body();
}


WorkerThread::WorkerThread(CPU *cpu)
	: _suspensionConditionVariable(), _cpu(cpu), _cpuToBeResumedOn(nullptr), _mustShutDown(false), _task(nullptr)
{
	int rc = pthread_create(&_pthread, &cpu->_pthreadAttr, &worker_thread_body_wrapper, this);
	FatalErrorHandler::handle(rc, " when creating a pthread in CPU ", cpu->_systemCPUId);
}


void *WorkerThread::body()
{
	ThreadManager::threadStartup(this);
	
	while (!_mustShutDown) {
		CPUActivation::activationCheck(this);
		
		if (_task == nullptr) {
			_task = Scheduler::getReadyTask(_cpu);
		} else {
			// The thread has been preassigned a task before being resumed
		}
		
		if (_task != nullptr) {
			WorkerThread *assignedThread = _task->getThread();
			
			// A task already assigned to another thread
			if (assignedThread != nullptr) {
				_task = nullptr;
				ThreadManager::addIdler(this);
				ThreadManager::switchThreads(this, assignedThread);
			} else {
				handleTask();
				_task = nullptr;
			}
		} else {
			// The code below is protected by a condition because under certain CPU activation/deactivation
			// cases, the call to CPUActivation::activationCheck may have put the thread in the idle queue
			// and the shutdown mechanism may have waken up the thread. In that case we do not want the
			// thread to go back to the idle queue. The previous case does not need the condition because
			// there is a task to be run and thus the program cannot be performing (a regular) shutdown.
			if (!_mustShutDown) {
				ThreadManager::addIdler(this);
				ThreadManager::switchThreads(this, nullptr);
			}
		}
	}
	
	ThreadManager::threadShutdownSequence(this);
	
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
		bool readyOrDisposable = _task->markAsFinished();
		Task *currentTask = _task;
		
		while ((currentTask != nullptr) && readyOrDisposable) {
			Task *parent = currentTask->getParent();
			
			if (currentTask->hasFinished()) {
				readyOrDisposable = currentTask->unlinkFromParent();
				// NOTE: The memory layout is defined in nanos_create_task
				currentTask->~Task();
				free(currentTask->getArgsBlock()); // FIXME: Need a proper object recycling mechanism here
				currentTask = parent;
			} else {
				// An ancestor in a taskwait that finishes at this point
				Scheduler::taskGetsUnblocked(currentTask, _cpu);
				readyOrDisposable = false;
			}
		}
	}
	
	_task = nullptr;
}

