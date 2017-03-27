#ifndef INSTRUMENT_USER_MUTEX_HPP
#define INSTRUMENT_USER_MUTEX_HPP


#include <InstrumentInstrumentationContext.hpp>


class UserMutex;


namespace Instrument {
	void acquiredUserMutex(UserMutex *userMutex, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	void blockedOnUserMutex(UserMutex *userMutex, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	void releasedUserMutex(UserMutex *userMutex, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
}


#endif // INSTRUMENT_USER_MUTEX_HPP
