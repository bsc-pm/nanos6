#ifndef INSTRUMENT_EXTRAE_TASK_EXECUTION_HPP
#define INSTRUMENT_EXTRAE_TASK_EXECUTION_HPP


#include "../InstrumentTaskExecution.hpp"
#include "InstrumentExtrae.hpp"
#include <InstrumentTaskId.hpp>

#include <executors/threads/WorkerThread.hpp>
#include <tasks/Task.hpp>


namespace Instrument {
	inline void startTask(task_id_t taskId, __attribute__((unused)) cpu_id_t cpuId, __attribute__((unused)) thread_id_t currentThreadId)
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
		
		ce.Types[1] = _codeLocation;
		ce.Values[1] = (extrae_value_t) taskId->run;
		
		Extrae_emit_CombinedEvents ( &ce );
	}
	
	
	inline void returnToTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) cpu_id_t cpuId, __attribute__((unused)) thread_id_t currentThreadId)
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
		
		Extrae_emit_CombinedEvents ( &ce );
	}
	
	
	inline void endTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) cpu_id_t cpuId, __attribute__((unused)) thread_id_t currentThreadId)
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
		
		ce.Types[1] = _codeLocation;
		ce.Values[1] = (extrae_value_t) nullptr;
		
		Extrae_emit_CombinedEvents ( &ce );
	}
	
	
	inline void destroyTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) cpu_id_t cpuId, __attribute__((unused)) thread_id_t currentThreadId)
	{
	}
}


#endif // INSTRUMENT_EXTRAE_TASK_EXECUTION_HPP
