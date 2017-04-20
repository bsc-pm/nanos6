#ifndef NAIVE_SCHEDULER_HPP
#define NAIVE_SCHEDULER_HPP


#include <deque>
#include <vector>

#include "../SchedulerInterface.hpp"
#include "lowlevel/SpinLock.hpp"
#include "executors/threads/CPU.hpp"


class Task;


class NaiveScheduler: public SchedulerInterface {
	SpinLock _globalLock;
	
	std::deque<Task *> _readyTasks;
	std::deque<Task *> _unblockedTasks;
	
	inline Task *getReplacementTask(CPU *computePlace);
	
public:
	NaiveScheduler();
	~NaiveScheduler();
	
	ComputePlace *addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint);
	
	void taskGetsUnblocked(Task *unblockedTask, ComputePlace *computePlace);
	
	Task *getReadyTask(ComputePlace *computePlace, Task *currentTask = nullptr, bool canMarkAsIdle = true);
	
	ComputePlace *getIdleComputePlace(bool force=false);
};


#endif // NAIVE_SCHEDULER_HPP

