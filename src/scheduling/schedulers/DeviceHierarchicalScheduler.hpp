#ifndef DEVICE_HIERARCHICAL_SCHEDULER_HPP
#define DEVICE_HIERARCHICAL_SCHEDULER_HPP

#include "../SchedulerInterface.hpp"


class Task;


class DeviceHierarchicalScheduler: public SchedulerInterface {
	SchedulerInterface *_CPUScheduler;

public:
	DeviceHierarchicalScheduler();
	~DeviceHierarchicalScheduler();
	
	ComputePlace *addReadyTask(Task *task, ComputePlace *hardwarePlace, ReadyTaskHint hint);
	
	void taskGetsUnblocked(Task *unblockedTask, ComputePlace *hardwarePlace);
	
	Task *getReadyTask(ComputePlace *hardwarePlace, Task *currentTask = nullptr);
	
	ComputePlace *getIdleComputePlace(bool force=false);
	
	void disableComputePlace(ComputePlace *hardwarePlace);
	
	void enableComputePlace(ComputePlace *hardwarePlace);

	static inline bool canBeCollapsed()
	{
		return true;
	}
};


#endif // DEVICE_HIERARCHICAL_SCHEDULER_HPP

