/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_TASK_WAIT_HPP
#define INSTRUMENT_TASK_WAIT_HPP


#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>


class Task;


namespace Instrument {
	void enterTaskWait(task_id_t taskId, char const *invocationSource, task_id_t if0TaskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	void exitTaskWait(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
}


#endif // INSTRUMENT_TASK_WAIT_HPP
