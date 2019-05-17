/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include "InstrumentDependenciesByGroup.hpp"
#include "InstrumentVerbose.hpp"

#include <InstrumentInstrumentationContext.hpp>


using namespace Instrument::Verbose;


namespace Instrument {
	void beginAccessGroup(
		__attribute__((unused)) task_id_t parentTaskId,
		__attribute__((unused)) void *handler,
		__attribute__((unused)) bool sequenceIsEmpty,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}


	void addTaskToAccessGroup(
		__attribute__((unused)) void *handler,
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}


	void removeTaskFromAccessGroup(
		__attribute__((unused)) void *handler,
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

}
