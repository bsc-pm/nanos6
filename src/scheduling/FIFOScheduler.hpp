#ifndef FIFO_SCHEDULER_HPP
#define FIFO_SCHEDULER_HPP


#include <deque>
#include <vector>

#include "SchedulerInterface.hpp"
#include "lowlevel/SpinLock.hpp"
#include "executors/threads/CPU.hpp"


class Task;


class FIFOScheduler: public SchedulerInterface {
	SpinLock _globalLock;
	
	std::deque<Task *> _readyTasks;
	std::deque<Task *> _unblockedTasks;
	
	std::deque<CPU *> _idleCPUs;
	
	inline CPU *getIdleCPU();
	inline Task *getReplacementTask(CPU *hardwarePlace);
	inline void cpuBecomesIdle(CPU *cpu);
	
public:
	FIFOScheduler();
	~FIFOScheduler();
	
	ComputePlace *addReadyTask(Task *task, ComputePlace *hardwarePlace, ReadyTaskHint hint);
	
	void taskGetsUnblocked(Task *unblockedTask, ComputePlace *hardwarePlace);
	
	Task *getReadyTask(ComputePlace *hardwarePlace, Task *currentTask = nullptr);
	
	ComputePlace *getIdleComputePlace(bool force=false);

    void addReadyQueue(std::size_t node_id);
};


#endif // FIFO_SCHEDULER_HPP

