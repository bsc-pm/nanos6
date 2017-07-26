/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_GRAPH_TASK_EXECUTION_HPP
#define INSTRUMENT_GRAPH_TASK_EXECUTION_HPP


#include "../api/InstrumentTaskExecution.hpp"


namespace Instrument {
	void startTask(task_id_t taskId, InstrumentationContext const &context);
	
	inline void returnToTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) InstrumentationContext const &context)
	{
	}
	
	void endTask(task_id_t taskId, InstrumentationContext const &context);
	
	inline void destroyTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) InstrumentationContext const &context)
	{
	}
}


#endif // INSTRUMENT_GRAPH_TASK_EXECUTION_HPP
