/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_FINALIZATION_HPP
#define TASK_FINALIZATION_HPP

class ComputePlace;
class Task;

class TaskFinalization {
public:
	static void taskFinished(Task *task, ComputePlace *computePlace, bool fromBusyThread = false);
	static void disposeTask(Task *task);
};


#endif // TASK_FINALIZATION_HPP
