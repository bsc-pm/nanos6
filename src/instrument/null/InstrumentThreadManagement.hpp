/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_NULL_THREAD_MANAGEMENT_HPP


#include "InstrumentExternalThreadId.hpp"
#include "InstrumentThreadId.hpp"
#include "../api/InstrumentThreadManagement.hpp"


namespace Instrument {
	inline void createdThread(/* OUT */ thread_id_t &threadId)
	{
		threadId = thread_id_t();
	}
	
	template<typename... TS>
	void createdExternalThread(/* OUT */ external_thread_id_t &threadId, __attribute__((unused)) TS... nameComponents)
	{
		threadId = external_thread_id_t();
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
