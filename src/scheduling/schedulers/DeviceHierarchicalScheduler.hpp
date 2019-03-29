/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef DEVICE_HIERARCHICAL_SCHEDULER_HPP
#define DEVICE_HIERARCHICAL_SCHEDULER_HPP

#include "../SchedulerInterface.hpp"


class Task;


class DeviceHierarchicalScheduler: public SchedulerInterface {
	SchedulerInterface *_CPUScheduler;

public:
	DeviceHierarchicalScheduler(int numaNodeIndex);
	~DeviceHierarchicalScheduler();
	
	ComputePlace *addReadyTask(Task *task, ComputePlace *hardwarePlace, ReadyTaskHint hint, bool doGetIdle = true);
	
	Task *getReadyTask(ComputePlace *hardwarePlace, Task *currentTask = nullptr, bool canMarkAsIdle = true, bool doWait = false);
	
	ComputePlace *getIdleComputePlace(bool force=false);
	
	void disableComputePlace(ComputePlace *hardwarePlace);
	
	void enableComputePlace(ComputePlace *hardwarePlace);
	
	bool requestPolling(ComputePlace *computePlace, polling_slot_t *pollingSlot, bool canMarkAsIdle = true);
	
	bool releasePolling(ComputePlace *computePlace, polling_slot_t *pollingSlot, bool canMarkAsIdle = true);
	
	std::string getName() const;
	
	static inline bool canBeCollapsed()
	{
		return true;
	}
};


#endif // DEVICE_HIERARCHICAL_SCHEDULER_HPP

