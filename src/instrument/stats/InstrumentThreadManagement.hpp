#ifndef INSTRUMENT_STATS_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_STATS_THREAD_MANAGEMENT_HPP


#include "../api/InstrumentThreadManagement.hpp"

#include "InstrumentStats.hpp"
#include "InstrumentThreadId.hpp"

#include "performance/HardwareCounters.hpp"


namespace Instrument {
	inline thread_id_t createdThread()
	{
		HardwareCounters::initializeThread();
		
		Stats::_threadStats = new Stats::ThreadInfo(true);
		
		Stats::_threadInfoListSpinLock.lock();
		Stats::_threadInfoList.push_back(Stats::_threadStats);
		Stats::_threadInfoListSpinLock.unlock();
		
		return thread_id_t();
	}
	
	inline void threadWillSuspend(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t computePlaceId)
	{
		Instrument::Stats::PhaseInfo &currentPhase = Stats::_threadStats->getCurrentPhaseRef();
		currentPhase._runningTime.continueAt(currentPhase._blockedTime);
	}
	
	inline void threadHasResumed(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t computePlaceId)
	{
		Instrument::Stats::PhaseInfo &currentPhase = Stats::_threadStats->getCurrentPhaseRef();
		currentPhase._blockedTime.continueAt(currentPhase._runningTime);
	}
	
}


#endif // INSTRUMENT_STATS_THREAD_MANAGEMENT_HPP
