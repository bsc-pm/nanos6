#include "system/ompss/UserMutex.hpp"

#include "ExecutionSteps.hpp"
#include "InstrumentUserMutex.hpp"
#include "InstrumentGraph.hpp"

#include "executors/threads/CPU.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "tasks/Task.hpp"

#include <cassert>


namespace Instrument {
	using namespace Graph;
	
	static inline usermutex_id_t getUserMutexId(UserMutex *userMutex, __attribute__((unused)) std::lock_guard<SpinLock> const &guard)
	{
		usermutex_id_t usermutexId;
		
		usermutex_to_id_map_t::iterator it = _usermutexToId.find(userMutex);
		if (it != _usermutexToId.end()) {
			usermutexId = it->second;
		} else {
			usermutexId = _nextUsermutexId++;
			_usermutexToId[userMutex] = usermutexId;
		}
		
		return usermutexId;
	}
	
	void acquiredUserMutex(UserMutex *userMutex)
	{
		std::lock_guard<SpinLock> guard(_graphLock);
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		if (currentThread == nullptr) {
			// The main task gets added by a non-worker thread
			// And other tasks can also be added by external threads in library mode
		}
		
		thread_id_t threadId = 0;
		if (currentThread != nullptr) {
			threadId = currentThread->getInstrumentationId();
		}
		
		long cpuId = -2;
		if (currentThread != nullptr) {
			CPU *cpu = currentThread->getComputePlace();
			assert(cpu != nullptr);
			cpuId = cpu->_virtualCPUId;
		}
		
		Task *task = currentThread->getTask();
		assert(task != nullptr);
		
		task_id_t taskId = task->getInstrumentationTaskId();
		
		usermutex_id_t usermutexId = getUserMutexId(userMutex, guard);
		
		enter_usermutex_step_t *enterUsermutexStep = new enter_usermutex_step_t(cpuId, threadId, usermutexId, taskId);
		_executionSequence.push_back(enterUsermutexStep);
	}
	
	void blockedOnUserMutex(UserMutex *userMutex)
	{
		std::lock_guard<SpinLock> guard(_graphLock);
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		if (currentThread == nullptr) {
			// The main task gets added by a non-worker thread
			// And other tasks can also be added by external threads in library mode
		}
		
		thread_id_t threadId = 0;
		if (currentThread != nullptr) {
			threadId = currentThread->getInstrumentationId();
		}
		
		long cpuId = -2;
		if (currentThread != nullptr) {
			CPU *cpu = currentThread->getComputePlace();
			assert(cpu != nullptr);
			cpuId = cpu->_virtualCPUId;
		}
		
		Task *task = currentThread->getTask();
		assert(task != nullptr);
		
		task_id_t taskId = task->getInstrumentationTaskId();
		
		usermutex_id_t usermutexId = getUserMutexId(userMutex, guard);
		
		block_on_usermutex_step_t *blockOnUsermutexStep = new block_on_usermutex_step_t(cpuId, threadId, usermutexId, taskId);
		_executionSequence.push_back(blockOnUsermutexStep);
	}
	
	void releasedUserMutex(UserMutex *userMutex)
	{
		std::lock_guard<SpinLock> guard(_graphLock);
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		if (currentThread == nullptr) {
			// The main task gets added by a non-worker thread
			// And other tasks can also be added by external threads in library mode
		}
		
		thread_id_t threadId = 0;
		if (currentThread != nullptr) {
			threadId = currentThread->getInstrumentationId();
		}
		
		long cpuId = -2;
		if (currentThread != nullptr) {
			CPU *cpu = currentThread->getComputePlace();
			assert(cpu != nullptr);
			cpuId = cpu->_virtualCPUId;
		}
		
		Task *task = currentThread->getTask();
		assert(task != nullptr);
		
		task_id_t taskId = task->getInstrumentationTaskId();
		
		usermutex_id_t usermutexId = getUserMutexId(userMutex, guard);
		
		exit_usermutex_step_t *exitUsermutexStep = new exit_usermutex_step_t(cpuId, threadId, usermutexId, taskId);
		_executionSequence.push_back(exitUsermutexStep);
	}
	
}

