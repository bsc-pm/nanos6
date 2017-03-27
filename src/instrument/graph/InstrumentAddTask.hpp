#ifndef INSTRUMENT_GRAPH_ADD_TASK_HPP
#define INSTRUMENT_GRAPH_ADD_TASK_HPP


#include "../api/InstrumentAddTask.hpp"

#include <InstrumentTaskId.hpp>


namespace Instrument {
	task_id_t enterAddTask(nanos_task_info *taskInfo, nanos_task_invocation_info *taskInvokationInfo, size_t flags);
	void createdTask(void *task, task_id_t taskId);
	void exitAddTask(task_id_t taskId);
}


#endif // INSTRUMENT_GRAPH_ADD_TASK_HPP
