/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_THREAD_MANAGEMENT_HPP

#include <string>

#include <InstrumentComputePlaceId.hpp>
#include <InstrumentExternalThreadId.hpp>
#include <InstrumentThreadId.hpp>


namespace Instrument {

	//! This function is called when the runtime creates a new thread and
	//! must return an instrumentation-specific thread identifier that will
	//! be used to identify it throughout the rest of the instrumentation API
	void enterThreadCreation(/* OUT */ thread_id_t &threadId, compute_place_id_t const &computePlaceId);
	void exitThreadCreation(thread_id_t threadId);

	void createdThread(thread_id_t threadId, compute_place_id_t const &computePlaceId);

	//! This function is called when the runtime creates a new non-worker thread and
	//! must return an instrumentation-specific thread identifier that will
	//! be used to identify it throughout the rest of the instrumentation API
	void precreatedExternalThread(/* OUT */ external_thread_id_t &threadId);

	template<typename... TS>
	void createdExternalThread(external_thread_id_t &threadId, TS... nameComponents);

	//! This function is called just after the initial synchronization,
	//! indicating that the thread has been granted permission to run after
	//! being created
	void threadSynchronizationCompleted(thread_id_t threadId);

	//! These functions are called when the thread suspends or resumes after
	//! the inital synchronization
	void threadWillSuspend(thread_id_t threadId, compute_place_id_t cpu, bool afterSynchronization = true);
	void threadHasResumed(thread_id_t threadId, compute_place_id_t cpu, bool afterSynchronization = true);

	void threadWillShutdown();
	void threadWillShutdown(external_thread_id_t threadId);

	void threadWillSuspend(external_thread_id_t threadId);
	void threadHasResumed(external_thread_id_t threadId);

	enum busy_wait_reason_t {
		scheduling_polling_slot_busy_wait_reason = 1
	};

	void threadEnterBusyWait(busy_wait_reason_t reason);
	void threadExitBusyWait();
}


#endif // INSTRUMENT_THREAD_MANAGEMENT_HPP
