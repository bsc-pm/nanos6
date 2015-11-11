#ifndef INSTRUMENT_NULL_DEPENDENCIES_BY_GROUP_HPP
#define INSTRUMENT_NULL_DEPENDENCIES_BY_GROUP_HPP


#include <InstrumentTaskId.hpp>

#include "../InstrumentDependenciesByGroup.hpp"


namespace Instrument {
	inline void beginAccessGroup(__attribute__((unused)) task_id_t parentTaskId, __attribute__((unused)) void *handler, __attribute__((unused)) bool sequenceIsEmpty)
	{
	}
	
	inline void addTaskToAccessGroup(__attribute__((unused)) void *handler, __attribute__((unused)) task_id_t taskId)
	{
	}
	
	inline void removeTaskFromAccessGroup(__attribute__((unused)) void *handler, __attribute__((unused)) task_id_t taskId)
	{
	}
	
}


#endif // INSTRUMENT_NULL_DEPENDENCIES_BY_GROUP_HPP
