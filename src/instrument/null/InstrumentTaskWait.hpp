#ifndef INSTRUMENT_NULL_TASK_WAIT_HPP
#define INSTRUMENT_NULL_TASK_WAIT_HPP


#include "../api/InstrumentTaskWait.hpp"
#include <InstrumentTaskExecution.hpp>

#include "executors/threads/CPU.hpp"
#include "executors/threads/WorkerThread.hpp"

#include <InstrumentTaskId.hpp>


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
		
		Instrument::returnToTask(taskId, cpuId, threadId);
	}
	
}


#endif // INSTRUMENT_NULL_TASK_WAIT_HPP
