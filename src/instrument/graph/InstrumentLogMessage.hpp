#ifndef INSTRUMENT_GRAPH_LOG_MESSAGE_HPP
#define INSTRUMENT_GRAPH_LOG_MESSAGE_HPP


#include <cassert>
#include <sstream>
#include <string>

#include "../api/InstrumentLogMessage.hpp"
#include "InstrumentGraph.hpp"
#include "InstrumentTaskId.hpp"

#include "ExecutionSteps.hpp"


using namespace Instrument::Graph;


namespace Instrument {
	namespace Graph {
		template<typename T>
		inline void fillStream(std::ostringstream &stream, T contents)
		{
			stream << contents;
		}
		
		template<typename T, typename... TS>
		inline void fillStream(std::ostringstream &stream, T content1, TS... contents)
		{
			stream << content1;
			fillStream(stream, contents...);
		}
	}
	
	
	template<typename... TS>
	inline void logMessage(InstrumentationContext const &context, TS... contents)
	{
		std::ostringstream stream;
		fillStream(stream, contents...);
		
		std::lock_guard<SpinLock> guard(_graphLock);
		log_message_step_t *step = new log_message_step_t(
			context._computePlaceId, context._threadId, context._taskId,
			stream.str()
		);
		_executionSequence.push_back(step);
		
	}
}


#endif // INSTRUMENT_GRAPH_LOG_MESSAGE_HPP
