#ifndef INSTRUMENT_GRAPH_TASK_WAIT_HPP
#define INSTRUMENT_GRAPH_TASK_WAIT_HPP


#include "../InstrumentTaskWait.hpp"

#include <InstrumentTaskId.hpp>


namespace Instrument {
	void enterTaskWait(task_id_t taskId, char const *invocationSource);
	void exitTaskWait(task_id_t taskId);
	
}


#endif // INSTRUMENT_GRAPH_TASK_WAIT_HPP
