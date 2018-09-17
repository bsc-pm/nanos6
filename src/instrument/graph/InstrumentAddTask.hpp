/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_GRAPH_ADD_TASK_HPP
#define INSTRUMENT_GRAPH_ADD_TASK_HPP


#include "../api/InstrumentAddTask.hpp"



namespace Instrument {
	task_id_t enterAddTask(nanos6_task_info_t *taskInfo, nanos6_task_invocation_info_t *taskInvokationInfo, size_t flags, InstrumentationContext const &context);
	void createdTask(void *task, task_id_t taskId, InstrumentationContext const &context);
	void exitAddTask(task_id_t taskId, InstrumentationContext const &context);
}


#endif // INSTRUMENT_GRAPH_ADD_TASK_HPP
