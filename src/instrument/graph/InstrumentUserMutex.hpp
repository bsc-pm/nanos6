/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_GRAPH_USER_MUTEX_HPP
#define INSTRUMENT_GRAPH_USER_MUTEX_HPP


#include "../api/InstrumentUserMutex.hpp"


namespace Instrument {
	void acquiredUserMutex(UserMutex *userMutex, InstrumentationContext const &context);
	void blockedOnUserMutex(UserMutex *userMutex, InstrumentationContext const &context);
	void releasedUserMutex(UserMutex *userMutex, InstrumentationContext const &context);

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


#endif // INSTRUMENT_GRAPH_USER_MUTEX_HPP
