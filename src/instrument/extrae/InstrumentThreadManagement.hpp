/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_EXTRAE_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_EXTRAE_THREAD_MANAGEMENT_HPP


#include "InstrumentExtrae.hpp"

#include "../api/InstrumentThreadManagement.hpp"
#include "../generic_ids/GenericIds.hpp"
#include "../support/InstrumentThreadLocalDataSupport.hpp"


// This is not defined in the extrae headers
extern "C" void Extrae_change_num_threads (unsigned n);


namespace Instrument {
	inline void createdThread(/* OUT */ thread_id_t &threadId)
	{
		ThreadLocalData &threadLocal = getThreadLocalData();
		threadLocal._nestingLevels.push_back(0);
		
		threadId = GenericIds::getNewThreadId();
		threadLocal._currentThreadId = threadId;
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.writeLock();
			Extrae_change_num_threads(extrae_nanos_get_num_threads());
			_extraeThreadCountLock.writeUnlock();
		}
		
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 1;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 1;
		ce.nCommunications = 0;
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		ce.Types[0] = _runtimeState;
		ce.Values[0] = (extrae_value_t) NANOS_IDLE;
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		Extrae_emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
	
	template<typename... TS>
	void createdExternalThread(/* OUT */ external_thread_id_t &threadId, __attribute__((unused)) TS... nameComponents)
	{
		ExternalThreadLocalData &threadLocal = getExternalThreadLocalData();
		
		if (_traceAsThreads) {
			// Same thread counter as regular worker threads
			threadId = GenericIds::getCommonPoolNewExternalThreadId();
		} else {
			// Conter separated from worker threads
			threadId = GenericIds::getNewExternalThreadId();
		}
		
		threadLocal._currentThreadId = threadId;
		
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 1;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 1;
		ce.nCommunications = 0;
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		ce.Types[0] = _runtimeState;
		ce.Values[0] = (extrae_value_t) NANOS_IDLE;
		
		_extraeThreadCountLock.writeLock();
		if (_traceAsThreads) {
			Extrae_change_num_threads(extrae_nanos_get_num_threads());
		} else {
			Extrae_change_num_threads(extrae_nanos_get_num_cpus_and_external_threads());
		}
		_extraeThreadCountLock.writeUnlock();
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		Extrae_emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
	
	inline void threadWillSuspend(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t cpu)
	{
	}
	
	inline void threadHasResumed(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t cpu)
	{
	}
	
	inline void threadWillShutdown()
	{
	}
	
	inline void threadEnterBusyWait(__attribute__((unused)) busy_wait_reason_t reason)
	{
	}
	
	inline void threadExitBusyWait()
	{
	}
}


#endif // INSTRUMENT_EXTRAE_THREAD_MANAGEMENT_HPP
