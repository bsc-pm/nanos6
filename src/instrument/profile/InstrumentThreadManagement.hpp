#ifndef INSTRUMENT_PROFILE_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_PROFILE_THREAD_MANAGEMENT_HPP


#include "../api/InstrumentThreadManagement.hpp"
#include "InstrumentProfile.hpp"


namespace Instrument {
	inline thread_id_t createdThread()
	{
		return Profile::createdThread();
	}
	
	inline void threadWillSuspend(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) cpu_id_t cpu)
	{
	}
	
	inline void threadHasResumed(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) cpu_id_t cpu)
	{
	}
	
}


#endif // INSTRUMENT_PROFILE_THREAD_MANAGEMENT_HPP
