/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_USER_MUTEX_HPP
#define INSTRUMENT_CTF_USER_MUTEX_HPP


#include "instrument/api/InstrumentUserMutex.hpp"
#include "CTFTracepoints.hpp"


namespace Instrument {
	inline void acquiredUserMutex(
		__attribute__((unused)) UserMutex *userMutex,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

	inline void blockedOnUserMutex(
		__attribute__((unused)) UserMutex *userMutex,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

	inline void releasedUserMutex(
		__attribute__((unused)) UserMutex *userMutex,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

	inline void enterUserMutexLock(
		__attribute__((unused)) InstrumentationContext const &context
	) {
		tp_mutex_lock_tc_enter();
	}

	inline void exitUserMutexLock(
		__attribute__((unused)) InstrumentationContext const &context
	) {
		tp_mutex_lock_tc_exit();
	}

	inline void enterUserMutexUnlock(
		__attribute__((unused)) InstrumentationContext const &context
	) {
		tp_mutex_unlock_tc_enter();
	}

	inline void exitUserMutexUnlock(
		__attribute__((unused)) InstrumentationContext const &context
	) {
		tp_mutex_unlock_tc_exit();
	}

}


#endif // INSTRUMENT_CTF_USER_MUTEX_HPP
