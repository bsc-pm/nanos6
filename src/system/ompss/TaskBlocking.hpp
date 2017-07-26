/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_BLOCKING_HPP
#define TASK_BLOCKING_HPP


class WorkerThread;
class Task;


class TaskBlocking {
public:
	static void taskBlocks(WorkerThread *currentThread, Task *currentTask, bool allowRunningInline);
};


#endif // TASK_BLOCKING_HPP
