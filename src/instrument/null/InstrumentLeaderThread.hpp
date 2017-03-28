#ifndef INSTRUMENT_NULL_LEADER_THREAD_HPP
#define INSTRUMENT_NULL_LEADER_THREAD_HPP


#include "../api/InstrumentLeaderThread.hpp"


namespace Instrument {
	inline void leaderThreadSpin(
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
}


#endif // INSTRUMENT_NULL_LEADER_THREAD_HPP
