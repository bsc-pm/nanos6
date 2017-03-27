#ifndef INSTRUMENT_TASK_WAIT_HPP
#define INSTRUMENT_TASK_WAIT_HPP


#include <InstrumentTaskId.hpp>


class Task;


namespace Instrument {
	void enterTaskWait(task_id_t taskId, char const *invocationSource);
	void exitTaskWait(task_id_t taskId);
}


#endif // INSTRUMENT_TASK_WAIT_HPP
