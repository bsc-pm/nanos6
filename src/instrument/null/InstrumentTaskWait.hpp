#ifndef INSTRUMENT_NULL_TASK_WAIT_HPP
#define INSTRUMENT_NULL_TASK_WAIT_HPP


#include "../api/InstrumentTaskWait.hpp"

#include <InstrumentTaskId.hpp>


namespace Instrument {
	inline void enterTaskWait(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) char const *invocationSource,
		__attribute__((unused)) task_id_t if0TaskId)
	{
	}
	
	inline void exitTaskWait(__attribute__((unused)) task_id_t taskId)
	{
	}
	
}


#endif // INSTRUMENT_NULL_TASK_WAIT_HPP
