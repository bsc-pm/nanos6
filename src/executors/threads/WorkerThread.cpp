#include "CPUActivation.hpp"
#include "ExternalThreadEnvironment.hpp"
#include "TaskFinalization.hpp"
#include "ThreadManager.hpp"
#include "WorkerThread.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/Task.hpp"

#include <DataAccessRegistration.hpp>

#include <InstrumentTaskExecution.hpp>
#include <InstrumentTaskStatus.hpp>

#include <atomic>

#include <pthread.h>


__thread WorkerThread *WorkerThread::_currentWorkerThread = nullptr;


static void *worker_thread_body_wrapper(void *parameter)
{
	WorkerThread *workerThread = (WorkerThread *) parameter;
	assert(workerThread != nullptr);
	return workerThread->body();
}


WorkerThread::WorkerThread(CPU *cpu)
	: _cpu(cpu), _cpuToBeResumedOn(nullptr), _mustShutDown(false)
{
	int rc = pthread_create(&_pthread, &cpu->_pthreadAttr, &worker_thread_body_wrapper, this);
	FatalErrorHandler::handle(rc, " when creating a pthread in CPU ", cpu->_systemCPUId);
}


void *WorkerThread::body()
{
	ThreadManager::threadStartup(this);
	
	while (!_mustShutDown) {
		if (!CPUActivation::acceptsWork(_cpu)) {
			Scheduler::disableHardwarePlace(_cpu);
			CPUActivation::activationCheck(this);
			Scheduler::enableHardwarePlace(_cpu);
		}
		
		if (_task == nullptr) {
			std::atomic<Task *> pollingSlot(nullptr);
			
			if (Scheduler::requestPolling(_cpu, &pollingSlot)) {
				while ((_task == nullptr) && !ThreadManager::mustExit() && CPUActivation::acceptsWork(_cpu)) {
					// Keep trying
					pollingSlot.compare_exchange_strong(_task, nullptr);
				}
				
				if (ThreadManager::mustExit()) {
					bool worked = Scheduler::releasePolling(_cpu, &pollingSlot);
					assert(worked && "A failure to release the scheduler polling slot means that the thread has got a task assigned, however the runtime is shutting down");
				}
				
				if (!CPUActivation::acceptsWork(_cpu)) {
					// The CPU is about to be disabled
					
					// Release the polling slot
					Scheduler::releasePolling(_cpu, &pollingSlot);
					
					// We may already have a task assigned through
					pollingSlot.compare_exchange_strong(_task, nullptr);
				}
			} else {
				// Did not receive neither the polling slot nor a task
			}
		} else {
			// The thread has been preassigned a task before being resumed
		}
		
		if (_task != nullptr) {
			EssentialThreadEnvironment *assignedThreadEnvironment = _task->getThread();
			
			// A task already assigned to another thread
			if (assignedThreadEnvironment != nullptr) {
				_task = nullptr;
				
				WorkerThread *assignedWorkerThread = dynamic_cast<WorkerThread *>(assignedThreadEnvironment);
				if (assignedWorkerThread != nullptr) {
					ThreadManager::addIdler(this);
					ThreadManager::switchThreads(this, assignedWorkerThread);
				} else {
					// The scheduler actually returned a task wrapping environment attached to an external
					// thread that was probably blocked on a taskwait
					#ifndef NDEBUG
						ExternalThreadEnvironment *externalThread = dynamic_cast<ExternalThreadEnvironment *>(assignedThreadEnvironment);
						assert(externalThread != nullptr);
					#endif
					
					// Wake it up and continue
					assignedThreadEnvironment->resume();
				}
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
	_task->setThread(this);
	
	Instrument::task_id_t taskId = _task->getInstrumentationTaskId();
	Instrument::startTask(taskId, _cpu->_virtualCPUId, _instrumentationId);
	Instrument::taskIsExecuting(taskId);
	
	// Run the task
	std::atomic_thread_fence(std::memory_order_acquire);
	_task->body();
	std::atomic_thread_fence(std::memory_order_release);
	
	Instrument::taskIsZombie(taskId);
	Instrument::endTask(taskId, _cpu->_virtualCPUId, _instrumentationId);
	
	// Release successors
	DataAccessRegistration::unregisterTaskDataAccesses(_task);
	
	if (_task->markAsFinished()) {
		TaskFinalization::disposeOrUnblockTask(_task, _cpu, this);
	}
	
	_task = nullptr;
}


