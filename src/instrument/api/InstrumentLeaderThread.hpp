#ifndef INSTRUMENT_LEADER_THREAD_HPP
#define INSTRUMENT_LEADER_THREAD_HPP

#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>


namespace Instrument {
	void leaderThreadSpin(InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
}


#endif // INSTRUMENT_LEADER_THREAD_HPP
