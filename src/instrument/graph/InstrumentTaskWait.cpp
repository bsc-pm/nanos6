#include "ExecutionSteps.hpp"
#include "InstrumentTaskWait.hpp"
#include "InstrumentGraph.hpp"

#include <InstrumentTaskExecution.hpp>

#include "executors/threads/CPU.hpp"
#include "executors/threads/WorkerThread.hpp"

#include <cassert>


namespace Instrument {
	using namespace Graph;
	
	
	void enterTaskWait(task_id_t taskId, char const *invocationSource)
	{
		std::lock_guard<SpinLock> guard(_graphLock);
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		
		thread_id_t threadId = 0;
		if (currentThread != nullptr) {
			threadId = currentThread->getInstrumentationId();
		}
		
		long cpuId = -2;
		if (currentThread != nullptr) {
			CPU *cpu = currentThread->getHardwarePlace();
			assert(cpu != nullptr);
			cpuId = cpu->_virtualCPUId;
		}
		
		taskwait_id_t taskwaitId = _nextTaskwaitId++;
		
		taskwait_t *taskwait = new taskwait_t(taskwaitId, invocationSource);
		enter_taskwait_step_t *enterTaskwaitStep = new enter_taskwait_step_t(cpuId, threadId, taskwaitId, taskId);
		
		task_info_t &taskInfo = _taskToInfoMap[taskId];
		
		taskInfo._phaseList.push_back(taskwait);
		
		_executionSequence.push_back(enterTaskwaitStep);
	}
	
	
	void exitTaskWait(task_id_t taskId)
	{
		std::lock_guard<SpinLock> guard(_graphLock);
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		
		thread_id_t threadId = 0;
		if (currentThread != nullptr) {
			threadId = currentThread->getInstrumentationId();
		}
		
		long cpuId = -2;
		if (currentThread != nullptr) {
			CPU *cpu = currentThread->getHardwarePlace();
			assert(cpu != nullptr);
			cpuId = cpu->_virtualCPUId;
		}
		
		task_info_t &taskInfo = _taskToInfoMap[taskId];
		
		assert(!taskInfo._phaseList.empty());
		phase_t *taskwaitPhase = taskInfo._phaseList.back();
		taskwait_t *taskwait = dynamic_cast<taskwait_t *> (taskwaitPhase);
		assert(taskwait != nullptr);
		taskwait_id_t taskwaitId = taskwait->_taskwaitId;
		
		exit_taskwait_step_t *exitTaskwaitStep = new exit_taskwait_step_t(cpuId, threadId, taskwaitId, taskId);
		_executionSequence.push_back(exitTaskwaitStep);
		
		// Instead of calling to Instrument::returnToTask we later on reuse the exitTaskwaitStep to also reactivate the task
	}
	
}
