/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_PROFILE_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_PROFILE_THREAD_MANAGEMENT_HPP


#include "../api/InstrumentThreadManagement.hpp"
#include "../support/InstrumentThreadLocalDataSupport.hpp"

#include "InstrumentProfile.hpp"



namespace Instrument {
	inline void createdThread(/* OUT */ thread_id_t &threadId, __attribute__((unused)) compute_place_id_t const &computePlaceId)
	{
		threadId = Profile::createdThread();
	}
	
	template<typename... TS>
	void createdExternalThread(/* OUT */ external_thread_id_t &threadId, __attribute__((unused)) TS... nameComponents)
	{
		// For now, external threads are not profiled
		threadId = external_thread_id_t();
		
		// Force the sentinel worker TLS to be initialized
		{
			ThreadLocalData &sentinelThreadLocal = getThreadLocalData();
			sentinelThreadLocal.init(Profile::getBufferSize());
		}
	}
	
	inline void threadWillSuspend(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t cpu)
	{
		Profile::disableForCurrentThread();
	}
	
	inline void threadHasResumed(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t cpu)
	{
		Profile::enableForCurrentThread();
	}
	
	inline void threadWillShutdown()
	{
		Profile::disableForCurrentThread();
	}
	
	inline void threadEnterBusyWait(__attribute__((unused)) busy_wait_reason_t reason)
	{
		Profile::disableForCurrentThread();
	}
	
	inline void threadExitBusyWait()
	{
		Profile::enableForCurrentThread();
	}
	
}


#endif // INSTRUMENT_PROFILE_THREAD_MANAGEMENT_HPP
