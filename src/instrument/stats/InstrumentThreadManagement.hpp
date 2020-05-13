/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_STATS_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_STATS_THREAD_MANAGEMENT_HPP

#include "InstrumentStats.hpp"
#include "InstrumentThreadId.hpp"
#include "InstrumentThreadLocalData.hpp"
#include "instrument/api/InstrumentThreadManagement.hpp"
#include "instrument/support/InstrumentThreadLocalDataSupport.hpp"


namespace Instrument {
	inline void enterThreadCreation(/* OUT */ thread_id_t &threadId, compute_place_id_t const &)
	{
		threadId = thread_id_t();
	}

	inline void exitThreadCreation(thread_id_t)
	{
	}

	inline void createdThread(thread_id_t, compute_place_id_t const &)
	{
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
	void createdExternalThread(external_thread_id_t &, TS...)
	{
	}

	inline void statsThreadWillSuspend()
	{
		ThreadLocalData &threadLocal = getThreadLocalData();

		Instrument::Stats::PhaseInfo &currentPhase = threadLocal._threadInfo.getCurrentPhaseRef();
		currentPhase._runningTime.continueAt(currentPhase._blockedTime);
	}

	inline void statsThreadHasResumed()
	{
		ThreadLocalData &threadLocal = getThreadLocalData();

		Instrument::Stats::PhaseInfo &currentPhase = threadLocal._threadInfo.getCurrentPhaseRef();
		currentPhase._blockedTime.continueAt(currentPhase._runningTime);
	}

	inline void threadWillSuspendBeforeSync(thread_id_t, compute_place_id_t)
	{
		statsThreadWillSuspend();
	}

	inline void threadHasResumedBeforeSync(thread_id_t, compute_place_id_t)
	{
		statsThreadHasResumed();
	}

	inline void threadWillSuspend(thread_id_t, compute_place_id_t)
	{
		statsThreadWillSuspend();
	}

	inline void threadHasResumed(thread_id_t, compute_place_id_t)
	{
		statsThreadHasResumed();
	}

	inline void threadWillSuspend(external_thread_id_t)
	{
	}

	inline void threadHasResumed(external_thread_id_t)
	{
	}

	inline void threadWillShutdown()
	{
	}

	inline void threadWillShutdown(__attribute__((unused)) external_thread_id_t threadId)
	{
	}

	inline void threadEnterBusyWait(busy_wait_reason_t)
	{
	}

	inline void threadExitBusyWait()
	{
	}
}


#endif // INSTRUMENT_STATS_THREAD_MANAGEMENT_HPP
