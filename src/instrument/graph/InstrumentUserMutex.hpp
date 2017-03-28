#ifndef INSTRUMENT_GRAPH_USER_MUTEX_HPP
#define INSTRUMENT_GRAPH_USER_MUTEX_HPP


#include "../api/InstrumentUserMutex.hpp"


namespace Instrument {
	void acquiredUserMutex(UserMutex *userMutex, InstrumentationContext const &context);
	void blockedOnUserMutex(UserMutex *userMutex, InstrumentationContext const &context);
	void releasedUserMutex(UserMutex *userMutex, InstrumentationContext const &context);
}


#endif // INSTRUMENT_GRAPH_USER_MUTEX_HPP
