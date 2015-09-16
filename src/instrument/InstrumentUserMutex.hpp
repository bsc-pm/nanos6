#ifndef INSTRUMENT_USER_MUTEX_HPP
#define INSTRUMENT_USER_MUTEX_HPP


class UserMutex;


namespace Instrument {
	void acquiredUserMutex(UserMutex *userMutex);
	void blockedOnUserMutex(UserMutex *userMutex);
	void releasedUserMutex(UserMutex *userMutex);
}


#endif // INSTRUMENT_USER_MUTEX_HPP
