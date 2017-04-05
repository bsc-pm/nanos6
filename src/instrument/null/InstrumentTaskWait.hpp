#ifndef INSTRUMENT_NULL_TASK_WAIT_HPP
#define INSTRUMENT_NULL_TASK_WAIT_HPP


#include "../api/InstrumentTaskWait.hpp"



namespace Instrument {
	inline void enterTaskWait(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) char const *invocationSource,
		__attribute__((unused)) task_id_t if0TaskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline void exitTaskWait(__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
}


#endif // INSTRUMENT_NULL_TASK_WAIT_HPP
