/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_TASK_EXECUTION_HPP
#define INSTRUMENT_TASK_EXECUTION_HPP


#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>

#include "InstrumentComputePlaceId.hpp"


namespace Instrument {
	void startTask(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	void returnToTask(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	void endTask(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	void destroyTask(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
}


#endif // INSTRUMENT_TASK_EXECUTION_HPP
