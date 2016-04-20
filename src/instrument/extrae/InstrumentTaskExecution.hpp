#ifndef INSTRUMENT_EXTRAE_TASK_EXECUTION_HPP
#define INSTRUMENT_EXTRAE_TASK_EXECUTION_HPP


#include "../InstrumentTaskExecution.hpp"
#include "InstrumentExtrae.hpp"
#include <InstrumentTaskId.hpp>



namespace Instrument {
	inline void startTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) CPU *cpu, __attribute__((unused)) WorkerThread *currentThread)
	{
      extrae_combined_events_t ce;

      ce.HardwareCounters = 0;
      ce.Callers = 0;
      ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
      ce.nEvents = 2;
      ce.nCommunications = 0;

      ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
      ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
 
      ce.Types[0] = _runtimeState;
      ce.Values[0] = (extrae_value_t) NANOS_RUNNING;

      nanos_task_info* TaskInfo =  currentThread->getTask()->getTaskInfo();
fprintf(stderr,"XTERUEL: executeTask: %s\n",_userFunctionMap[(void *) TaskInfo->run]);

      ce.Types[1] = _codeLocation;
      ce.Values[1] = (extrae_value_t) TaskInfo->run;


fprintf(stderr,"Extrae_emit_CombinedEvents %s:%d:%d\n","start task",_runtimeState, NANOS_RUNNING); //FIXME
      Extrae_emit_CombinedEvents ( &ce );

	}
	
	inline void returnToTask(__attribute__((unused)) task_id_t taskIdk, __attribute__((unused)) CPU *cpu, __attribute__((unused)) WorkerThread *currentThread)
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

fprintf(stderr,"Extrae_emit_CombinedEvents %s:%d:%d\n","return task",_runtimeState, NANOS_NO_STATE); //FIXME
      Extrae_emit_CombinedEvents ( &ce );

	}
	
	inline void endTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) CPU *cpu, __attribute__((unused)) WorkerThread *currentThread)
	{
      extrae_combined_events_t ce;

      ce.HardwareCounters = 0;
      ce.Callers = 0;
      ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
      ce.nEvents = 2;
      ce.nCommunications = 0;

      ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
      ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
 
      ce.Types[0] = _runtimeState;
      ce.Values[0] = (extrae_value_t) NANOS_NO_STATE;

fprintf(stderr,"Extrae_emit_CombinedEvents %s:%d:%d\n","End task",_runtimeState, NANOS_NO_STATE); //FIXME

      ce.Types[1] = _codeLocation;
      ce.Values[1] = (extrae_value_t) nullptr;

      Extrae_emit_CombinedEvents ( &ce );
	}
	
	inline void destroyTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) CPU *cpu, __attribute__((unused)) WorkerThread *currentThread)
	{
	}
}


#endif // INSTRUMENT_EXTRAE_TASK_EXECUTION_HPP
