#include "CPUActivation.hpp"
#include "TaskFinalization.hpp"
#include "ThreadManager.hpp"
#include "WorkerThread.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/Task.hpp"
#include "memory/cache/GenericCache.hpp"
#include "hardware/places/MemoryPlace.hpp"
#include "memory/directory/Directory.hpp"

#include <DataAccessRegistration.hpp>

#include <InstrumentTaskExecution.hpp>
#include <InstrumentTaskStatus.hpp>

#include <atomic>

#include <pthread.h>
#include <cstring>

#define _unused(x) ((void)(x))

__thread WorkerThread *WorkerThread::_currentWorkerThread = nullptr;


WorkerThread::WorkerThread(CPU *cpu)
	: _cpu(cpu), _cpuToBeResumedOn(nullptr), _mustShutDown(false)
{
	start(&cpu->_pthreadAttr);
}


void *WorkerThread::body()
{
	ThreadManager::threadStartup(this);
	
	_cpu->bindThread(_tid);
	
	while (!_mustShutDown) {
		CPUActivation::activationCheck(this);
		
		if (_task == nullptr) {
			std::atomic<Task *> pollingSlot(nullptr);
			
			if (Scheduler::requestPolling(_cpu, &pollingSlot)) {
				while ((_task == nullptr) && !ThreadManager::mustExit() && CPUActivation::acceptsWork(_cpu)) {
					// Keep trying
					pollingSlot.compare_exchange_strong(_task, nullptr);
				}
				
				if (ThreadManager::mustExit()) {
					__attribute__((unused)) bool worked = Scheduler::releasePolling(_cpu, &pollingSlot);
					assert(worked && "A failure to release the scheduler polling slot means that the thread has got a task assigned, however the runtime is shutting down");
                    _unused(worked);
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
    if(_task->hasEnabledCopies() && _task->hasPendingCopies()) {
        // task is preReady
        //! Do some data transferences if any
        float * cachesLoad = nullptr;
        GenericCache * destCache = _task->getCache();
        if(destCache == nullptr) {
            destCache = _cpu->getMemoryPlaces()[0]->getCache();
            _task->setCache(destCache);
        }
        assert(destCache != nullptr);
        destCache->copyData(cachesLoad, _task);
    }
    else {
        // task is ready
        _task->setThread(this);
        
        //! Temporal print to check that each task is executing where it should according to its data affinity.
        //std::cerr << "Task with label " << _task->getTaskInfo()->task_label << " is executed on NUMA node " 
        //    << _cpu->_NUMANodeId << ". It's data size is:" << _task->getDataSize() << "." << std::endl;

        //! Notify CacheTrackingSet the task data.
        Directory::registerLastLevelCacheData(_task->getDataAccesses(), _cpu->_NUMANodeId, _task);

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
        // Release copies
        GenericCache * destCache = _task->getCache();
        if(destCache != nullptr)
            destCache->releaseCopies(_task);

        if (_task->markAsFinished()) {
            TaskFinalization::disposeOrUnblockTask(_task, _cpu, this);
        }
    }

	_task = nullptr;
}


