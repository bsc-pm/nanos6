#ifndef INSTRUMENT_EXTRAE_TASK_WAIT_HPP
#define INSTRUMENT_EXTRAE_TASK_WAIT_HPP


#include "../api/InstrumentTaskWait.hpp"
#include <InstrumentTaskExecution.hpp>

#include "InstrumentExtrae.hpp"


namespace Instrument {
	inline void enterTaskWait(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) char const *invocationSource,
		__attribute__((unused)) task_id_t if0TaskId,
		__attribute__((unused)) InstrumentationContext const &context)
	{
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 1;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 1;
		ce.nCommunications = 0;
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		ce.Types[0] = _runtimeState;
		ce.Values[0] = (extrae_value_t) NANOS_SYNCHRONIZATION;
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		Extrae_emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
	
	
	inline void exitTaskWait(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context)
	{
		returnToTask(taskId, context);
	}
	
}

#endif // INSTRUMENT_EXTRAE_TASK_WAIT_HPP
