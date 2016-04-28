#ifndef INSTRUMENT_EXTRAE_TASK_STATUS_HPP
#define INSTRUMENT_EXTRAE_TASK_STATUS_HPP


#include "../InstrumentTaskStatus.hpp"
#include <InstrumentTaskId.hpp>


namespace Instrument {
	inline void taskIsPending(__attribute__((unused)) task_id_t taskId)
	{
	}
	
	inline void taskIsReady(__attribute__((unused)) task_id_t taskId)
	{
	}
	
	inline void taskIsExecuting(__attribute__((unused)) task_id_t taskId)
	{
	}
	
	inline void taskIsBlocked(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) task_blocking_reason_t reason)
	{
	}
	
	inline void taskIsZombie(__attribute__((unused)) task_id_t taskId)
	{
	}
	
}


#endif // INSTRUMENT_EXTRAE_TASK_STATUS_HPP
