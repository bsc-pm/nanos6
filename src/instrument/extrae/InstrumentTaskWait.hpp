#ifndef INSTRUMENT_EXTRAE_TASK_WAIT_HPP
#define INSTRUMENT_EXTRAE_TASK_WAIT_HPP


#include "../InstrumentTaskWait.hpp"
#include <InstrumentTaskExecution.hpp>

#include "executors/threads/CPU.hpp"
#include "executors/threads/WorkerThread.hpp"

#include "InstrumentExtrae.hpp"
#include <InstrumentTaskId.hpp>


namespace Instrument {
	inline void enterTaskWait(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) char const *invocationSource)
	{
      extrae_combined_events_t ce;

      ce.HardwareCounters = 0;
      ce.Callers = 0;
      ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
      ce.nEvents = 1;
      ce.nCommunications = 0;

      ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
      ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
 
      ce.Types[0] = _runtimeState;
      ce.Values[0] = (extrae_value_t) NANOS_SYNCHRONIZATION;

//fprintf(stderr,"Extrae_emit_CombinedEvents %s:%d:%d\n","enter task wait",_runtimeState, NANOS_SYNCHRONIZATION); //FIXME:debug
      Extrae_emit_CombinedEvents ( &ce );
	}
	
	void exitTaskWait(task_id_t taskId)
	{
      extrae_combined_events_t ce;

      ce.HardwareCounters = 0;
      ce.Callers = 0;
      ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
      ce.nEvents = 1;
      ce.nCommunications = 0;

      ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
      ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
 
      ce.Types[0] = _runtimeState;
      ce.Values[0] = (extrae_value_t) NANOS_NO_STATE;

//fprintf(stderr,"Extrae_emit_CombinedEvents %s:%d:%d\n","exit task wait",_runtimeState, NANOS_NO_STATE); //FIXME:debug
      Extrae_emit_CombinedEvents ( &ce );

//TODO: Emit returnToTask at this point
#if 0
		WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
		assert(thread != nullptr);
		
		CPU *cpu = (CPU *) thread->getHardwarePlace();
		assert(cpu != nullptr);
		
		Instrument::returnToTask(taskId, cpu, thread);
#endif
	}
	
}

#endif // INSTRUMENT_EXTRAE_TASK_WAIT_HPP
