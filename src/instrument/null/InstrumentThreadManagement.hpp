#ifndef INSTRUMENT_NULL_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_NULL_THREAD_MANAGEMENT_HPP


#include "InstrumentThreadId.hpp"
#include "../api/InstrumentThreadManagement.hpp"


namespace Instrument {
	inline thread_id_t createdThread()
	{
		return thread_id_t();
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


#endif // INSTRUMENT_NULL_THREAD_MANAGEMENT_HPP
