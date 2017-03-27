#ifndef INSTRUMENT_TASK_WAIT_HPP
#define INSTRUMENT_TASK_WAIT_HPP


#include <InstrumentInstrumentationContext.hpp>


class Task;


namespace Instrument {
	void enterTaskWait(task_id_t taskId, char const *invocationSource, task_id_t if0TaskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	void exitTaskWait(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
}


#endif // INSTRUMENT_TASK_WAIT_HPP
