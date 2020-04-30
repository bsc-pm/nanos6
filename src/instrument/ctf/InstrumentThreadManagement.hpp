/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_CTF_THREAD_MANAGEMENT_HPP


#include "InstrumentExternalThreadId.hpp"
#include "InstrumentThreadId.hpp"
#include "../api/InstrumentThreadManagement.hpp"

#include "ctfapi/CTFTypes.hpp"


namespace Instrument {

	ctf_thread_id_t gettid(void);

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
		ctf_thread_id_t tid = gettid();

		threadId = external_thread_id_t(tid);
	}

	template<typename... TS>
	void createdExternalThread(__attribute__((unused))  external_thread_id_t &threadId, __attribute__((unused)) TS... nameComponents)
	{
	}

	inline void threadWillSuspend(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t cpu)
	{
	}

	inline void threadHasResumed(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t cpu)
	{
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

	inline void threadEnterBusyWait(__attribute__((unused)) busy_wait_reason_t reason)
	{
	}

	inline void threadExitBusyWait()
	{
	}
}


#endif // INSTRUMENT_CTF_THREAD_MANAGEMENT_HPP
