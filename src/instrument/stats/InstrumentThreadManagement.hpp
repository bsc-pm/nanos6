/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_STATS_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_STATS_THREAD_MANAGEMENT_HPP


#include "InstrumentHardwareCounters.hpp"
#include "InstrumentStats.hpp"
#include "InstrumentThreadId.hpp"
#include "InstrumentThreadLocalData.hpp"
#include "../api/InstrumentThreadManagement.hpp"
#include "../support/InstrumentThreadLocalDataSupport.hpp"


namespace Instrument {
	inline void enterThreadCreation(/* OUT */ thread_id_t &threadId, __attribute__((unused)) compute_place_id_t const &computePlaceId)
	{
		threadId = thread_id_t();
	}
	
	inline void exitThreadCreation(__attribute__((unused)) thread_id_t threadId)
	{
	}
	
	inline void createdThread(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t const &computePlaceId)
	{
		InstrumentHardwareCounters::initializeThread();
		
		ThreadLocalData &threadLocal = getThreadLocalData();
		
		Stats::_threadInfoListSpinLock.lock();
		Stats::_threadInfoList.push_back(&threadLocal._threadInfo);
		Stats::_threadInfoListSpinLock.unlock();
	}
	
	inline void precreatedExternalThread(/* OUT */ external_thread_id_t &threadId)
	{
		// For now, external threads are not instrumented
		threadId = external_thread_id_t();
	}
	
	template<typename... TS>
	void createdExternalThread(__attribute__((unused)) external_thread_id_t &threadId, __attribute__((unused)) TS... nameComponents)
	{
	}
	
	inline void threadWillSuspend(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t computePlaceId)
	{
		ThreadLocalData &threadLocal = getThreadLocalData();
		
		Instrument::Stats::PhaseInfo &currentPhase = threadLocal._threadInfo.getCurrentPhaseRef();
		currentPhase._runningTime.continueAt(currentPhase._blockedTime);
	}
	
	inline void threadHasResumed(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t computePlaceId)
	{
		ThreadLocalData &threadLocal = getThreadLocalData();
		
		Instrument::Stats::PhaseInfo &currentPhase = threadLocal._threadInfo.getCurrentPhaseRef();
		currentPhase._blockedTime.continueAt(currentPhase._runningTime);
	}
	
	inline void threadWillSuspend(__attribute__((unused)) external_thread_id_t threadId)
	{
	}
	
	inline void threadHasResumed(__attribute__((unused)) external_thread_id_t threadId)
	{
	}
	
	inline void threadWillShutdown()
	{
		// Clean PAPI events for the current thread
		InstrumentHardwareCounters::shutdownThread();
	}
	
	inline void threadEnterBusyWait(__attribute__((unused)) busy_wait_reason_t reason)
	{
	}
	
	inline void threadExitBusyWait()
	{
	}
}


#endif // INSTRUMENT_STATS_THREAD_MANAGEMENT_HPP
