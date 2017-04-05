#ifndef INSTRUMENT_TASK_EXECUTION_HPP
#define INSTRUMENT_TASK_EXECUTION_HPP


#include <InstrumentInstrumentationContext.hpp>

#include "InstrumentComputePlaceId.hpp"


namespace Instrument {
	void startTask(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	void returnToTask(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	void endTask(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	void destroyTask(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
}


#endif // INSTRUMENT_TASK_EXECUTION_HPP
