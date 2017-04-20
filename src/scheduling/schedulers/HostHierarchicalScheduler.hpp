#ifndef HOST_HIERARCHICAL_SCHEDULER_HPP
#define HOST_HIERARCHICAL_SCHEDULER_HPP

#include "../SchedulerInterface.hpp"


class Task;


class HostHierarchicalScheduler: public SchedulerInterface {
	SchedulerInterface *_NUMAScheduler;

public:
	HostHierarchicalScheduler();
	~HostHierarchicalScheduler();
	
	ComputePlace *addReadyTask(Task *task, ComputePlace *hardwarePlace, ReadyTaskHint hint, bool doGetIdle = true);
	
	void taskGetsUnblocked(Task *unblockedTask, ComputePlace *hardwarePlace);
	
	Task *getReadyTask(ComputePlace *hardwarePlace, Task *currentTask = nullptr, bool canMarkAsIdle = true);
	
	ComputePlace *getIdleComputePlace(bool force=false);
	
	void disableComputePlace(ComputePlace *hardwarePlace);
	
	void enableComputePlace(ComputePlace *hardwarePlace);

	static inline bool canBeCollapsed()
	{
		return true;
	}
};


#endif // HOST_HIERARCHICAL_SCHEDULER_HPP

