/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_GRAPH_TASK_WAIT_HPP
#define INSTRUMENT_GRAPH_TASK_WAIT_HPP


#include "../api/InstrumentTaskWait.hpp"

#include <InstrumentInstrumentationContext.hpp>


namespace Instrument {
	void enterTaskWait(task_id_t taskId, char const *invocationSource, task_id_t if0TaskId, InstrumentationContext const &context);
	void exitTaskWait(task_id_t taskId, InstrumentationContext const &context);
	
}


#endif // INSTRUMENT_GRAPH_TASK_WAIT_HPP
