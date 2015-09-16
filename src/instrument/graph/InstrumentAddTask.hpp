#ifndef INSTRUMENT_GRAPH_ADD_TASK_HPP
#define INSTRUMENT_GRAPH_ADD_TASK_HPP


#include "../InstrumentAddTask.hpp"

#include <InstrumentTaskId.hpp>


class Task;


namespace Instrument {
	task_id_t enterAddTask(nanos_task_info *taskInfo, nanos_task_invocation_info *taskInvokationInfo);
	void createdTask(Task *task, task_id_t taskId);
	void exitAddTask(task_id_t taskId);
}


#endif // INSTRUMENT_GRAPH_ADD_TASK_HPP
