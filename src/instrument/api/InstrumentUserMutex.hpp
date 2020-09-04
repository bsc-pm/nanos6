/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_USER_MUTEX_HPP
#define INSTRUMENT_USER_MUTEX_HPP


#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>


class UserMutex;


namespace Instrument {
	//! This function is called when the current worker thread has aquried a user mutex
	//! \param[in] userMutex The user mutex pointer
	void acquiredUserMutex(UserMutex *userMutex, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());

	//! This function is called when the current worker thread blocks on a user mutex
	//! \param[in] userMutex The user mutex pointer
	void blockedOnUserMutex(UserMutex *userMutex, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());

	//! This function is called when the current worker thread releases a previously acquired user mutex
	//! \param[in] userMutex The user mutex pointer
	void releasedUserMutex(UserMutex *userMutex, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());

	//! This function is called when entering the user mutex lock Nanos6 API
	//! Task Hardware Counters are always updated before calling this function
	void enterUserMutexLock(InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());

	//! This function is called when exiting the user mutex lock Nanos6 API.
	//! Runtime Hardware Counters are always updated before calling this function
	void exitUserMutexLock(InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());

	//! This function is called when entering the user mutex unlock Nanos6 API
	//! Task Hardware Counters are always updated before calling this function
	void enterUserMutexUnlock(InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());

	//! This function is called when exiting the user mutex unlock Nanos6 API.
	//! Runtime Hardware Counters are always updated before calling this function
	void exitUserMutexUnlock(InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
}


#endif // INSTRUMENT_USER_MUTEX_HPP
