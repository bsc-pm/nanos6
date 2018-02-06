/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_BLOCKING_HPP
#define INSTRUMENT_BLOCKING_HPP


#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>

#include <InstrumentTaskId.hpp>


namespace Instrument {
	void enterBlocking(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	void exitBlocking(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	void unblockTask(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
}


#endif // INSTRUMENT_BLOCKING_HPP
