#ifndef INSTRUMENT_GRAPH_TASK_WAIT_HPP
#define INSTRUMENT_GRAPH_TASK_WAIT_HPP


#include "../api/InstrumentTaskWait.hpp"

#include <InstrumentInstrumentationContext.hpp>


namespace Instrument {
	void enterTaskWait(task_id_t taskId, char const *invocationSource, task_id_t if0TaskId, InstrumentationContext const &context);
	void exitTaskWait(task_id_t taskId, InstrumentationContext const &context);
	
}


#endif // INSTRUMENT_GRAPH_TASK_WAIT_HPP
