/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_USER_MUTEX_HPP
#define INSTRUMENT_NULL_USER_MUTEX_HPP


#include "instrument/api/InstrumentUserMutex.hpp"


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
	}

	inline void exitUserMutexLock(
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

	inline void enterUserMutexUnlock(
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

	inline void exitUserMutexUnlock(
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

}


#endif // INSTRUMENT_NULL_USER_MUTEX_HPP
