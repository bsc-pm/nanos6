/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2022 Barcelona Supercomputing Center (BSC)
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
	//! param[in] threadId The thread identifier
	//! param[in] cpu The compute place where the current thread is bound
	void enterThreadCreation(
		/* OUT */ thread_id_t &threadId,
		compute_place_id_t const &computePlaceId
	);

	//! This function is called when the runtime has finished creating the
	//! new thread
	//! param[in] threadId The thread identifier
	void exitThreadCreation(thread_id_t threadId);

	//! This function is called by a newly created worker thread on wake up.
	//! At this point, the thread does not own the cpu object and might
	//! oversubscribe with its parent
	//! param[in] threadId The thread identifier
	//! param[in] cpu The compute place where the current thread is bound
	void createdThread(
		thread_id_t threadId,
		compute_place_id_t const &computePlaceId
	);

	//! This function is called when the runtime creates an external thread thread and
	//! must return an instrumentation-specific thread identifier that will
	//! be used to identify it throughout the rest of the instrumentation API
	//! param[in] threadId The thread identifier
	void precreatedExternalThread(/* OUT */ external_thread_id_t &threadId);

	//! This function is called as the last step of creating an external thread
	//! param[in] threadId The thread identifier
	//! param[in] nameComponents The thread human-readable name
	template<typename... TS>
	void createdExternalThread(external_thread_id_t &threadId, TS... nameComponents);

	//! This function is called just after the initial synchronization,
	//! indicating that the thread has been granted permission to run after
	//! being created
	void threadSynchronizationCompleted(thread_id_t threadId);

	//! This function is called when the current worker thread suspends.
	//! Runtime Hardware Counters are always updated before calling this function
	//! param[in] threadId The thread identifier
	//! param[in] cpu The compute place where the current thread is bound
	//! param[in] afterSynchronization Whether the current thread has
	//! performed the initial synchronization (i.e. whether it owns the
	//! current cpu or not)
	void threadWillSuspend(
		thread_id_t threadId,
		compute_place_id_t cpu,
		bool afterSynchronization = true
	);

	//! This function is called when the current worker thread resumes
	//! param[in] threadId The thread identifier
	//! param[in] cpu The compute place where the current thread is bound
	//! param[in] afterSynchronization Whether the current thread has
	//! performed the initial synchronization (i.e. whether it owns the
	//! current cpu or not)
	void threadHasResumed(
		thread_id_t threadId,
		compute_place_id_t cpu,
		bool afterSynchronization = true
	);

	//! This function is called when the current worker threads is about to shutdown
	//! Runtime Hardware Counters are always updated before calling this function
	void threadWillShutdown();

	//! This function is called when the current external threads is about to shutdown
	//! \param[in] threadId The thread identifier
	void threadWillShutdown(external_thread_id_t threadId);

	//! This function is called when the current external threads is about to suspend.
	//! \param[in] threadId The thread identifier
	void threadWillSuspend(external_thread_id_t threadId);

	//! This function is called when the current external threads has resumed
	//! \param[in] threadId The thread identifier
	void threadHasResumed(external_thread_id_t threadId);
}


#endif // INSTRUMENT_THREAD_MANAGEMENT_HPP
