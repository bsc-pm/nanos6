/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_NULL_THREAD_MANAGEMENT_HPP


#include "InstrumentExternalThreadId.hpp"
#include "InstrumentThreadId.hpp"
#include "instrument/api/InstrumentThreadManagement.hpp"


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
	}

	inline void precreatedExternalThread(/* OUT */ external_thread_id_t &threadId)
	{
		threadId = external_thread_id_t();
	}

	template<typename... TS>
	void createdExternalThread(__attribute__((unused))  external_thread_id_t &threadId, __attribute__((unused)) TS... nameComponents)
	{
	}

	inline void threadSynchronizationCompleted(
		__attribute((unused)) thread_id_t threadId
	) {
	}

	inline void threadWillSuspend(
		__attribute__((unused)) thread_id_t threadId,
		__attribute__((unused)) compute_place_id_t cpu,
		__attribute__((unused)) bool afterSynchronization
	) {
	}

	inline void threadHasResumed(
		__attribute__((unused)) thread_id_t threadId,
		__attribute__((unused)) compute_place_id_t cpu,
		__attribute__((unused)) bool afterSynchronization
	) {
	}

	inline void threadWillSuspend(__attribute__((unused)) external_thread_id_t threadId)
	{
	}

	inline void threadHasResumed(__attribute__((unused)) external_thread_id_t threadId)
	{
	}

	inline void threadWillShutdown()
	{
	}

	inline void threadWillShutdown(__attribute__((unused)) external_thread_id_t threadId)
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
