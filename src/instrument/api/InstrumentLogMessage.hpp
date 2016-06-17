#ifndef INSTRUMENT_LOG_MESSAGE_HPP
#define INSTRUMENT_LOG_MESSAGE_HPP


#include <InstrumentTaskId.hpp>


namespace Instrument {
	template<typename... TS>
	void logMessage(task_id_t triggererTaskId, TS... components);
}


#endif // INSTRUMENT_LOG_MESSAGE_HPP
