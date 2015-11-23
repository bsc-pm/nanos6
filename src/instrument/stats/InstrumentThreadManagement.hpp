#ifndef INSTRUMENT_STATS_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_STATS_THREAD_MANAGEMENT_HPP


#include "../InstrumentThreadManagement.hpp"

#include "InstrumentStats.hpp"


namespace Instrument {
	inline void createdThread(__attribute__((unused)) WorkerThread *thread)
	{
		Stats::_threadStats = new Stats::ThreadInfo(true);
		
		Stats::_threadInfoListSpinLock.lock();
		Stats::_threadInfoList.push_back(Stats::_threadStats);
		Stats::_threadInfoListSpinLock.unlock();
	}
	
	inline void threadWillSuspend(__attribute__((unused)) WorkerThread *thread, __attribute__((unused)) CPU *cpu)
	{
		Stats::_threadStats->_runningTime.continueAt(Stats::_threadStats->_blockedTime);
	}
	
	inline void threadHasResumed(__attribute__((unused)) WorkerThread *thread, __attribute__((unused)) CPU *cpu)
	{
		Stats::_threadStats->_blockedTime.continueAt(Stats::_threadStats->_runningTime);
	}
	
}


#endif // INSTRUMENT_STATS_THREAD_MANAGEMENT_HPP
