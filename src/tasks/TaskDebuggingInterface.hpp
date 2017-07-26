/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

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
