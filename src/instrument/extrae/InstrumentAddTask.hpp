/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_EXTRAE_ADD_TASK_HPP
#define INSTRUMENT_EXTRAE_ADD_TASK_HPP


#include "../api/InstrumentAddTask.hpp"
#include "../support/InstrumentThreadLocalDataSupport.hpp"
#include "InstrumentExtrae.hpp"

#include <cassert>
#include <mutex>


class Task;


namespace Instrument {
	inline task_id_t enterAddTask(
		nanos_task_info *taskInfo,
		__attribute__((unused)) nanos_task_invocation_info *taskInvokationInfo,
		__attribute__((unused)) size_t flags,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 0;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 1;
		ce.nCommunications = 0;
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		ce.Types[0] = _runtimeState;
		ce.Values[0] = (extrae_value_t) NANOS_CREATION;
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		Extrae_emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
		
		assert(taskInfo != nullptr);
		{
			std::lock_guard<SpinLock> guard(_userFunctionMapLock);
			_userFunctionMap.insert(taskInfo);
		}
		
		ThreadLocalData &threadLocal = getThreadLocalData();
		if (threadLocal._nestingLevels.empty()) {
			// This may be an external thread, therefore assume that it is a spawned task
			return task_id_t(taskInfo, 0);
		}
		
		return task_id_t(taskInfo, threadLocal._nestingLevels.back()+1);
	}
	
	
	inline void createdTask(__attribute__((unused)) void *task, __attribute__((unused)) task_id_t taskId, __attribute__((unused)) InstrumentationContext const &context)
	{
	}
	
	
	inline void exitAddTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) InstrumentationContext const &context)
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
		ce.Values[0] = (extrae_value_t) NANOS_RUNNING;
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		Extrae_emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
}


#endif // INSTRUMENT_EXTRAE_ADD_TASK_HPP
