#ifndef INSTRUMENT_GRAPH_TASK_WAIT_HPP
#define INSTRUMENT_GRAPH_TASK_WAIT_HPP


#include "../api/InstrumentTaskWait.hpp"

#include <InstrumentTaskId.hpp>


namespace Instrument {
	void enterTaskWait(task_id_t taskId, char const *invocationSource, task_id_t if0TaskId);
	void exitTaskWait(task_id_t taskId);
	
}


#endif // INSTRUMENT_GRAPH_TASK_WAIT_HPP
