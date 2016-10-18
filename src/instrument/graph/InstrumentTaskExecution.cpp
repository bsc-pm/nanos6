#include "ExecutionSteps.hpp"
#include "InstrumentTaskExecution.hpp"
#include "InstrumentGraph.hpp"

#include "executors/threads/WorkerThread.hpp"
#include "tasks/Task.hpp"

#include <cassert>
#include <mutex>


namespace Instrument {
	using namespace Graph;
	
	
	void startTask(task_id_t taskId, cpu_id_t cpuId, thread_id_t currentThreadId)
	{
		std::lock_guard<SpinLock> guard(_graphLock);
		enter_task_step_t *enterTaskStep = new enter_task_step_t(cpuId, currentThreadId, taskId);
		_executionSequence.push_back(enterTaskStep);
	}
	
	void endTask(__attribute__((unused)) task_id_t taskId, cpu_id_t cpuId, thread_id_t currentThreadId)
	{
		std::lock_guard<SpinLock> guard(_graphLock);
		exit_task_step_t *exitTaskStep = new exit_task_step_t(cpuId, currentThreadId, taskId);
		_executionSequence.push_back(exitTaskStep);
	}
	
}
