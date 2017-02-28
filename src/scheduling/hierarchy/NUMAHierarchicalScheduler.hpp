#ifndef NUMA_HIERARCHICAL_SCHEDULER_HPP
#define NUMA_HIERARCHICAL_SCHEDULER_HPP


#include <vector>

#include "../SchedulerInterface.hpp"
#include "lowlevel/SpinLock.hpp"


class Task;


class NUMAHierarchicalScheduler: public SchedulerInterface {
	std::vector<SchedulerInterface *> _NUMANodeScheduler;
	
	// Use atomics to avoid using a lock
	// Be careful, as std::atomic does not have a copy operation, many vector
	// operations are not usable
	std::vector<std::atomic<int>> _readyTasks;

public:
	NUMAHierarchicalScheduler();
	~NUMAHierarchicalScheduler();

	ComputePlace *addReadyTask(Task *task, ComputePlace *hardwarePlace, ReadyTaskHint hint);
	
	void taskGetsUnblocked(Task *unblockedTask, ComputePlace *hardwarePlace);
	
	Task *getReadyTask(ComputePlace *hardwarePlace, Task *currentTask = nullptr);
	
	ComputePlace *getIdleComputePlace(bool force=false);
};


#endif // NUMA_HIERARCHICAL_SCHEDULER_HPP

