#ifndef TASK_DEBUGGING_INTERFACE_HPP
#define TASK_DEBUGGING_INTERFACE_HPP

#include "Task.hpp"


class TaskDebuggingInterface {
public:
	static Task *getRuntimeTask(void *taskHandle)
	{
		return (Task *) taskHandle;
	}
};


#endif // TASK_DEBUGGING_INTERFACE_HPP
