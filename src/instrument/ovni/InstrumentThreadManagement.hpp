/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_OVNI_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_OVNI_THREAD_MANAGEMENT_HPP


#include "InstrumentExternalThreadId.hpp"
#include "InstrumentThreadId.hpp"
#include "instrument/api/InstrumentThreadManagement.hpp"
#include "../support/InstrumentThreadLocalDataSupport.hpp"


namespace Instrument {
	inline void enterThreadCreation(__attribute__((unused)) /* OUT */ thread_id_t &threadId, __attribute__((unused)) compute_place_id_t const &computePlaceId)
	{
	}

	inline void exitThreadCreation(__attribute__((unused)) thread_id_t threadId)
	{
	}

	void createdThread(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t const &computePlaceId);
	void precreatedExternalThread(/* OUT */ external_thread_id_t &threadId);

	template<typename... TS>
	void createdExternalThread(__attribute__((unused)) external_thread_id_t &threadId, __attribute__((unused)) TS... nameComponents)
	{
	}

	void threadSynchronizationCompleted(thread_id_t threadId);
	void threadWillSuspend(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t cpu, bool afterSynchronization);
	void threadHasResumed(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t cpu, bool afterSynchronization);
	void threadWillSuspend(external_thread_id_t threadId);
	void threadHasResumed(external_thread_id_t threadId);
	void threadWillShutdown();
	void threadWillShutdown(external_thread_id_t threadId);
}


#endif // INSTRUMENT_OVNI_THREAD_MANAGEMENT_HPP
