#ifndef FIFO_SCHEDULER_HPP
#define FIFO_SCHEDULER_HPP


#include <deque>
#include <vector>

#include "../SchedulerInterface.hpp"
#include "lowlevel/SpinLock.hpp"
#include "executors/threads/CPU.hpp"


class Task;


class FIFOScheduler: public SchedulerInterface {
	SpinLock _globalLock;
	
	std::deque<Task *> _readyTasks;
	std::deque<Task *> _unblockedTasks;
	
	inline Task *getReplacementTask(CPU *computePlace);
	
public:
	FIFOScheduler();
	~FIFOScheduler();
	
	ComputePlace *addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint, bool doGetIdle = true);
	
	void taskGetsUnblocked(Task *unblockedTask, ComputePlace *computePlace);
	
	Task *getReadyTask(ComputePlace *computePlace, Task *currentTask = nullptr, bool canMarkAsIdle = true);
	
	ComputePlace *getIdleComputePlace(bool force=false);
	
	std::string getName() const;
};


#endif // FIFO_SCHEDULER_HPP

