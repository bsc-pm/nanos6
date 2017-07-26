/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_DEPENDENCIES_BY_GROUP_HPP
#define INSTRUMENT_NULL_DEPENDENCIES_BY_GROUP_HPP



#include "../api/InstrumentDependenciesByGroup.hpp"


namespace Instrument {
	inline void beginAccessGroup(
		__attribute__((unused)) task_id_t parentTaskId,
		__attribute__((unused)) void *handler,
		__attribute__((unused)) bool sequenceIsEmpty,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline void addTaskToAccessGroup(
		__attribute__((unused)) void *handler,
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline void removeTaskFromAccessGroup(
		__attribute__((unused)) void *handler,
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
}


#endif // INSTRUMENT_NULL_DEPENDENCIES_BY_GROUP_HPP
