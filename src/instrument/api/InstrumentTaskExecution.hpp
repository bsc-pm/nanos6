/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_TASK_EXECUTION_HPP
#define INSTRUMENT_TASK_EXECUTION_HPP


#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>

#include "InstrumentComputePlaceId.hpp"


namespace Instrument {

	//! This function is called just before start executing a task
	//! Task Hardware Counters are always updated before calling this function
	//! \param[in] taskid The task identifier
	void startTask(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());

	//! This function is called just after a task has finished
	//! Runtime Hardware Counters are always updated before calling this function
	//! \param[in] taskid The task identifier
	void endTask(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());

	//! This function is called when tasks resources are no longer needed
	//! \param[in] taskid The task identifier
	void destroyTask(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
}


#endif // INSTRUMENT_TASK_EXECUTION_HPP
