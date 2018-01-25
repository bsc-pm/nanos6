/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_EXTRAE_TASK_EXECUTION_HPP
#define INSTRUMENT_EXTRAE_TASK_EXECUTION_HPP


#include "../api/InstrumentTaskExecution.hpp"
#include "../support/InstrumentThreadLocalDataSupport.hpp"
#include "InstrumentExtrae.hpp"

#include <cassert>


namespace Instrument {
	inline void startTask(
		task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 1;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 4;
		ce.nCommunications = 0;
		
		if (!_sampleTaskCount) {
			ce.nEvents += 1;
		}
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		ce.Types[0] = _runtimeState;
		ce.Values[0] = (extrae_value_t) NANOS_RUNNING;
		
		ce.Types[1] = _codeLocation;
		ce.Values[1] = (extrae_value_t) taskId._taskInfo->run;
		
		ce.Types[2] = _nestingLevel;
		ce.Values[2] = (extrae_value_t) taskId._nestingLevel;
		
		ce.Types[3] = _taskInstanceId;
		ce.Values[3] = (extrae_value_t) taskId._taskId;
		
		size_t readyTasks = --_readyTasks;
		if (!_sampleTaskCount) {
			ce.Types[4] = _readyTasksEventType;
			ce.Values[4] = (extrae_value_t) readyTasks;
			
			// This counter is not so reliable, so try to skip underflows
			if (((signed long long) ce.Values[4]) < 0) {
				ce.Values[4] = 0;
			}
		}
		
		ThreadLocalData &threadLocal = getThreadLocalData();
		threadLocal._nestingLevels.push_back(taskId._nestingLevel);
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		Extrae_emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
	
	
	inline void returnToTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 1;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 4;
		ce.nCommunications = 0;
		
		if (!_sampleTaskCount) {
			ce.nEvents += 1;
		}
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		ce.Types[0] = _runtimeState;
		ce.Values[0] = (extrae_value_t) NANOS_RUNNING;
		
		ce.Types[1] = _codeLocation;
		ce.Values[1] = (extrae_value_t) taskId._taskInfo->run;
		
		ce.Types[2] = _nestingLevel;
		ce.Values[2] = (extrae_value_t) taskId._nestingLevel;
		
		ce.Types[3] = _taskInstanceId;
		ce.Values[3] = (extrae_value_t) taskId._taskId;
		
		size_t readyTasks = --_readyTasks;
		if (!_sampleTaskCount) {
			ce.Types[4] = _readyTasksEventType;
			ce.Values[4] = (extrae_value_t) readyTasks;
			
			// This counter is not so reliable, so try to skip underflows
			if (((signed long long) ce.Values[4]) < 0) {
				ce.Values[4] = 0;
			}
		}
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		Extrae_emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
	
	
	inline void endTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 1;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 4;
		ce.nCommunications = 0;
		
		if (!_sampleTaskCount) {
			ce.nEvents += 1;
		}
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		ce.Types[0] = _runtimeState;
		ce.Values[0] = (extrae_value_t) NANOS_IDLE;
		
		ce.Types[1] = _codeLocation;
		ce.Values[1] = (extrae_value_t) nullptr;
		
		ce.Types[2] = _nestingLevel;
		ce.Values[2] = (extrae_value_t) nullptr;
		
		ce.Types[3] = _taskInstanceId;
		ce.Values[3] = (extrae_value_t) nullptr;
		
		size_t liveTasks = --_liveTasks;
		if (!_sampleTaskCount) {
			ce.Types[4] = _liveTasksEventType;
			ce.Values[4] = (extrae_value_t) liveTasks;
			
			// This counter is not so reliable, so try to skip underflows
			if (((signed long long) ce.Values[4]) < 0) {
				ce.Values[4] = 0;
			}
		}
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		Extrae_emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
		
		ThreadLocalData &threadLocal = getThreadLocalData();
		assert(!threadLocal._nestingLevels.empty());
		threadLocal._nestingLevels.pop_back();
	}
	
	
	inline void destroyTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
}


#endif // INSTRUMENT_EXTRAE_TASK_EXECUTION_HPP
