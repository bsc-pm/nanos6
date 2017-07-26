/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_GRAPH_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_GRAPH_THREAD_MANAGEMENT_HPP


#include <InstrumentComputePlaceId.hpp>
#include <InstrumentThreadId.hpp>

#include "../api/InstrumentThreadManagement.hpp"
#include "../generic_ids/GenericIds.hpp"


namespace Instrument {
	inline thread_id_t createdThread()
	{
		return GenericIds::getNewThreadId();
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


#endif // INSTRUMENT_GRAPH_THREAD_MANAGEMENT_HPP
