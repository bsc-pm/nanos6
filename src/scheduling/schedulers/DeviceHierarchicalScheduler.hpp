#ifndef DEVICE_HIERARCHICAL_SCHEDULER_HPP
#define DEVICE_HIERARCHICAL_SCHEDULER_HPP

#include "../SchedulerInterface.hpp"


class Task;


class DeviceHierarchicalScheduler: public SchedulerInterface {
	SchedulerInterface *_CPUScheduler;

public:
	DeviceHierarchicalScheduler(int nodeIndex);
	~DeviceHierarchicalScheduler();
	
	ComputePlace *addReadyTask(Task *task, ComputePlace *hardwarePlace, ReadyTaskHint hint, bool doGetIdle = true);
	
	void taskGetsUnblocked(Task *unblockedTask, ComputePlace *hardwarePlace);
	
	Task *getReadyTask(ComputePlace *hardwarePlace, Task *currentTask = nullptr, bool canMarkAsIdle = true);
	
	ComputePlace *getIdleComputePlace(bool force=false);
	
	void disableComputePlace(ComputePlace *hardwarePlace);
	
	void enableComputePlace(ComputePlace *hardwarePlace);
	
	bool requestPolling(ComputePlace *computePlace, polling_slot_t *pollingSlot);
	
	bool releasePolling(ComputePlace *computePlace, polling_slot_t *pollingSlot);
	
	std::string getName() const;
	
	static inline bool canBeCollapsed()
	{
		return true;
	}
};


#endif // DEVICE_HIERARCHICAL_SCHEDULER_HPP

