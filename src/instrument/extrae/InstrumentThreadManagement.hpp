#ifndef INSTRUMENT_EXTRAE_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_EXTRAE_THREAD_MANAGEMENT_HPP


#include "InstrumentExtrae.hpp"

#include "../generic_ids/GenericIds.hpp"
#include "../api/InstrumentThreadManagement.hpp"


// This is not defined in the extrae headers
extern "C" void Extrae_change_num_threads (unsigned n);


namespace Instrument {
	inline thread_id_t createdThread()
	{
		_nestingLevels.push_back(0);
		
		thread_id_t threadId = GenericIds::getNewThreadId();
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.writeLock();
			
			_currentThreadId = new thread_id_t(threadId);
			Extrae_change_num_threads(extrae_nanos_get_num_threads());
			
			_extraeThreadCountLock.writeUnlock();
		}
		
		return threadId;
	}
	
	inline void threadWillSuspend(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) hardware_place_id_t cpu)
	{
	}
	
	inline void threadHasResumed(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) hardware_place_id_t cpu)
	{
	}
}


#endif // INSTRUMENT_EXTRAE_THREAD_MANAGEMENT_HPP
