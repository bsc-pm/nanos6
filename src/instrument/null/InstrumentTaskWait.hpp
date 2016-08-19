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
	
	void exitTaskWait(task_id_t taskId)
	{
		WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
		assert(thread != nullptr);
		
		CPU *cpu = (CPU *) thread->getHardwarePlace();
		assert(cpu != nullptr);
		
		Instrument::returnToTask(taskId, cpu->_virtualCPUId, thread->getInstrumentationId());
	}
	
}


#endif // INSTRUMENT_NULL_TASK_WAIT_HPP
