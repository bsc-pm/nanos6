/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_USER_MUTEX_HPP
#define INSTRUMENT_USER_MUTEX_HPP


#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>


class UserMutex;


namespace Instrument {
	void acquiredUserMutex(UserMutex *userMutex, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	void blockedOnUserMutex(UserMutex *userMutex, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	void releasedUserMutex(UserMutex *userMutex, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
}


#endif // INSTRUMENT_USER_MUTEX_HPP
