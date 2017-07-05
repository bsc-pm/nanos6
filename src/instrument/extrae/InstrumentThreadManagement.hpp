#ifndef INSTRUMENT_EXTRAE_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_EXTRAE_THREAD_MANAGEMENT_HPP


#include "InstrumentExtrae.hpp"

#include "../api/InstrumentThreadManagement.hpp"
#include "../generic_ids/GenericIds.hpp"
#include "../support/InstrumentThreadLocalDataSupport.hpp"


// This is not defined in the extrae headers
extern "C" void Extrae_change_num_threads (unsigned n);


namespace Instrument {
	inline thread_id_t createdThread()
	{
		ThreadLocalData &threadLocal = getThreadLocalData();
		threadLocal._nestingLevels.push_back(0);
		
		thread_id_t threadId = GenericIds::getNewThreadId();
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.writeLock();
			
			threadLocal._currentThreadId = new thread_id_t(threadId);
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
		
		return threadId;
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
}


#endif // INSTRUMENT_EXTRAE_THREAD_MANAGEMENT_HPP
