#include "InstrumentTaskExecution.hpp"
#include "InstrumentGraph.hpp"

#include "executors/threads/WorkerThread.hpp"
#include "tasks/Task.hpp"

#include <cassert>
#include <mutex>


namespace Instrument {
	using namespace Graph;
	
	
	void startTask(task_id_t taskId, CPU *cpu, WorkerThread *currentThread)
	{
		std::lock_guard<SpinLock> guard(_graphLock);
		assert(cpu != nullptr);
		
		assert(_threadToId.find(currentThread) != _threadToId.end());
		thread_id_t threadId = _threadToId[currentThread];
		
		enter_task_step_t *enterTaskStep = new enter_task_step_t(cpu->_virtualCPUId, threadId, taskId);
		_executionSequence.push_back(enterTaskStep);
	}
	
	void returnToTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) CPU *cpu,
		__attribute__((unused)) WorkerThread *currentThread)
	{
	}
	
	void endTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) CPU *cpu, WorkerThread *currentThread)
	{
		std::lock_guard<SpinLock> guard(_graphLock);
		assert(cpu != nullptr);
		
		assert(_threadToId.find(currentThread) != _threadToId.end());
		thread_id_t threadId = _threadToId[currentThread];
		
		exit_task_step_t *exitTaskStep = new exit_task_step_t(cpu->_virtualCPUId, threadId, taskId);
		_executionSequence.push_back(exitTaskStep);
	}
	
}
