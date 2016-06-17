#ifndef INSTRUMENT_STATS_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_STATS_THREAD_MANAGEMENT_HPP


#include "../InstrumentThreadManagement.hpp"

#include "InstrumentStats.hpp"
#include "InstrumentThreadId.hpp"


namespace Instrument {
	inline thread_id_t createdThread()
	{
		Stats::_threadStats = new Stats::ThreadInfo(true);
		
		Stats::_threadInfoListSpinLock.lock();
		Stats::_threadInfoList.push_back(Stats::_threadStats);
		Stats::_threadInfoListSpinLock.unlock();
		
		return thread_id_t();
	}
	
	inline void threadWillSuspend(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) cpu_id_t cpuId)
	{
		Stats::_threadStats->_runningTime.continueAt(Stats::_threadStats->_blockedTime);
	}
	
	inline void threadHasResumed(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) cpu_id_t cpuId)
	{
		Stats::_threadStats->_blockedTime.continueAt(Stats::_threadStats->_runningTime);
	}
	
}


#endif // INSTRUMENT_STATS_THREAD_MANAGEMENT_HPP
