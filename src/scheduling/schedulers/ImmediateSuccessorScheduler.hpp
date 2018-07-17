/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef IMMEDIATE_SUCCESSOR_SCHEDULER_HPP
#define IMMEDIATE_SUCCESSOR_SCHEDULER_HPP


#include <deque>
#include <vector>

#include "../SchedulerInterface.hpp"
#include "lowlevel/SpinLock.hpp"
#include "executors/threads/CPU.hpp"


class Task;


class ImmediateSuccessorScheduler: public SchedulerInterface {
	SpinLock _globalLock;
	
	std::deque<Task *> _readyTasks;
	std::deque<Task *> _unblockedTasks;
	
	inline Task *getReplacementTask(CPU *computePlace);
	
public:
	ImmediateSuccessorScheduler(int numaNodeIndex);
	~ImmediateSuccessorScheduler();
	
	ComputePlace *addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint, bool doGetIdle = true);
	
	Task *getReadyTask(ComputePlace *computePlace, Task *currentTask = nullptr, bool canMarkAsIdle = true, bool doWait = false);
	
	ComputePlace *getIdleComputePlace(bool force=false);
	
	void disableComputePlace(ComputePlace *computePlace);
	
	std::string getName() const;
};


#endif // IMMEDIATE_SUCCESSOR_SCHEDULER_HPP

