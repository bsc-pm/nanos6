/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_EXTRAE_TASK_WAIT_HPP
#define INSTRUMENT_EXTRAE_TASK_WAIT_HPP


#include "../api/InstrumentTaskWait.hpp"
#include "../support/InstrumentThreadLocalDataSupport.hpp"
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
		ce.nEvents = 4;
		ce.nCommunications = 0;
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		ce.Types[0] = _runtimeState;
		ce.Values[0] = (extrae_value_t) NANOS_SYNCHRONIZATION;
		
		ce.Types[1] = _codeLocation;
		ce.Values[1] = (extrae_value_t) (extrae_value_t) nullptr;
		
		ce.Types[2] = _nestingLevel;
		ce.Values[2] = (extrae_value_t) (extrae_value_t) nullptr;
		
		ce.Types[3] = _taskInstanceId;
		ce.Values[3] = (extrae_value_t) (extrae_value_t) nullptr;
		
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
