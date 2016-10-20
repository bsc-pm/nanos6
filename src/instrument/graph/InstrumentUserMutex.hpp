#ifndef INSTRUMENT_GRAPH_USER_MUTEX_HPP
#define INSTRUMENT_GRAPH_USER_MUTEX_HPP


#include "../api/InstrumentUserMutex.hpp"


namespace Instrument {
	void acquiredUserMutex(UserMutex *userMutex);
	void blockedOnUserMutex(UserMutex *userMutex);
	void releasedUserMutex(UserMutex *userMutex);
}


#endif // INSTRUMENT_GRAPH_USER_MUTEX_HPP
