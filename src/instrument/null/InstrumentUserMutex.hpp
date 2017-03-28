#ifndef INSTRUMENT_NULL_USER_MUTEX_HPP
#define INSTRUMENT_NULL_USER_MUTEX_HPP


#include "../api/InstrumentUserMutex.hpp"


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
	
}


#endif // INSTRUMENT_NULL_USER_MUTEX_HPP
