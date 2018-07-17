/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef NAIVE_GPU_SCHEDULER_HPP
#define NAIVE_GPU_SCHEDULER_HPP

#include <deque>
#include <vector>

#include "scheduling/SchedulerInterface.hpp"
#include "lowlevel/SpinLock.hpp"

class Task;
class CUDAComputePlace;

class CUDANaiveScheduler: public SchedulerInterface {
	SpinLock _globalLock;
	
	//CUDA Tasks
	std::deque<Task *> _readyTasks;
	std::deque<Task *> _blockedTasks; 	
	
	//CUDA devices
	std::deque<CUDAComputePlace *> _idleGpus;
	
public:
	CUDANaiveScheduler(int numaNodeIndex);
	~CUDANaiveScheduler();
	
	ComputePlace *addReadyTask(Task *task, ComputePlace *hardwarePlace, ReadyTaskHint hint, bool doGetIdle = true);
	
	Task *getReadyTask(ComputePlace *hardwarePlace, Task *currentTask = nullptr, bool canMarkAsIdle = true, bool doWait = false);
	
	ComputePlace *getIdleComputePlace(bool force=false);
	
	std::string getName() const;
};


#endif // NAIVE_GPU_SCHEDULER_HPP

