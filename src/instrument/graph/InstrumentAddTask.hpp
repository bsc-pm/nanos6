#ifndef INSTRUMENT_GRAPH_ADD_TASK_HPP
#define INSTRUMENT_GRAPH_ADD_TASK_HPP


#include "../api/InstrumentAddTask.hpp"



namespace Instrument {
	task_id_t enterAddTask(nanos_task_info *taskInfo, nanos_task_invocation_info *taskInvokationInfo, size_t flags, InstrumentationContext const &context);
	void createdTask(void *task, task_id_t taskId, InstrumentationContext const &context);
	void exitAddTask(task_id_t taskId, InstrumentationContext const &context);
}


#endif // INSTRUMENT_GRAPH_ADD_TASK_HPP
