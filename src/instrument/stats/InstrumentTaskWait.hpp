#ifndef INSTRUMENT_STATS_TASK_WAIT_HPP
#define INSTRUMENT_STATS_TASK_WAIT_HPP


#include "../api/InstrumentTaskWait.hpp"

#include "executors/threads/CPU.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "tasks/Task.hpp"

#include "InstrumentTaskExecution.hpp"
#include "InstrumentTaskId.hpp"
#include "InstrumentStats.hpp"

#include <atomic>


namespace Instrument {
	inline void enterTaskWait(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) char const *invocationSource)
	{
	}
	
	inline void exitTaskWait(task_id_t taskId)
	{
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		if (currentThread == nullptr) {
			// The main task gets added by a non-worker thread
			// And other tasks can also be added by external threads in library mode
		}
		
		thread_id_t threadId;
		if (currentThread != nullptr) {
			threadId = currentThread->getInstrumentationId();
		}
		
		long cpuId = -2;
		if (currentThread != nullptr) {
			CPU *cpu = currentThread->getComputePlace();
			assert(cpu != nullptr);
			cpuId = cpu->_virtualCPUId;
		}
		
		// If a spawned function, count the taskwait as a frontier between phases
		Task *task = currentThread->getTask();
		assert(task != nullptr);
		if (task->getParent() == nullptr) {
			assert(Instrument::Stats::_currentPhase == (Instrument::Stats::_phaseTimes.size() - 1));
			Instrument::Stats::_phaseTimes.back().stop();
			Instrument::Stats::_phaseTimes.emplace_back(true);
			
			Instrument::Stats::_currentPhase++;
		}
		
		Instrument::returnToTask(taskId, cpuId, threadId);
	}
	
}


#endif // INSTRUMENT_STATS_TASK_WAIT_HPP
