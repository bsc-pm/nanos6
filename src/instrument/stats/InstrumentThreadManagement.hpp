#ifndef INSTRUMENT_STATS_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_STATS_THREAD_MANAGEMENT_HPP


#include "../api/InstrumentThreadManagement.hpp"
#include "../support/InstrumentThreadLocalDataSupport.hpp"

#include "InstrumentStats.hpp"
#include "InstrumentThreadId.hpp"

#include "performance/HardwareCounters.hpp"


namespace Instrument {
	inline thread_id_t createdThread()
	{
		HardwareCounters::initializeThread();
		
		ThreadLocalData &threadLocal = getThreadLocalData();
		
		Stats::_threadInfoListSpinLock.lock();
		Stats::_threadInfoList.push_back(&threadLocal._threadInfo);
		Stats::_threadInfoListSpinLock.unlock();
		
		return thread_id_t();
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
	
	inline void threadWillShutdown()
	{
		// Clean PAPI events for the current thread
		HardwareCounters::shutdownThread();
	}
	
	inline void threadEnterBusyWait(__attribute__((unused)) busy_wait_reason_t reason)
	{
	}
	
	inline void threadExitBusyWait()
	{
	}
}


#endif // INSTRUMENT_STATS_THREAD_MANAGEMENT_HPP
