#ifndef INSTRUMENT_NULL_LOG_MESSAGE_HPP
#define INSTRUMENT_NULL_LOG_MESSAGE_HPP


#include "../api/InstrumentLogMessage.hpp"


namespace Instrument {
	template<typename... TS>
	inline void logMessage(__attribute__((unused)) task_id_t triggererTaskId, __attribute__((unused)) TS... components)
	{
	}
}


#endif // INSTRUMENT_NULL_LOG_MESSAGE_HPP
