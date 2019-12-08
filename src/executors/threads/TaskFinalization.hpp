/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_FINALIZATION_HPP
#define TASK_FINALIZATION_HPP

class ComputePlace;
class Task;

class TaskFinalization {
public:
	static void disposeOrUnblockTask(Task *task, ComputePlace *computePlace, bool fromBusyThread = false);

};


#endif // TASK_FINALIZATION_HPP
