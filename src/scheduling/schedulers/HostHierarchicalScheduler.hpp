/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef HOST_HIERARCHICAL_SCHEDULER_HPP
#define HOST_HIERARCHICAL_SCHEDULER_HPP

#include "../SchedulerInterface.hpp"


class Task;


class HostHierarchicalScheduler: public SchedulerInterface {
	SchedulerInterface *_NUMAScheduler;
	SchedulerInterface *_CUDAScheduler;
	
public:
	HostHierarchicalScheduler();
	~HostHierarchicalScheduler();
	
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


#endif // HOST_HIERARCHICAL_SCHEDULER_HPP

