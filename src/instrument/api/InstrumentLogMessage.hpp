#ifndef INSTRUMENT_LOG_MESSAGE_HPP
#define INSTRUMENT_LOG_MESSAGE_HPP


#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>


namespace Instrument {
	template<typename... TS>
	void logMessage(InstrumentationContext const &context /* = ThreadInstrumentationContext::getCurrent() */, TS... components);
}


#endif // INSTRUMENT_LOG_MESSAGE_HPP
