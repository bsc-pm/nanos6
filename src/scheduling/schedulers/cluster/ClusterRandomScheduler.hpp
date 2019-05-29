/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef CLUSTER_RANDOM_SCHEDULER_HPP
#define CLUSTER_RANDOM_SCHEDULER_HPP

#include "scheduling/SchedulerInterface.hpp"


class Task;
class ClusterNode;


class ClusterRandomScheduler: public SchedulerInterface {
	SchedulerInterface *_hostScheduler;
	ClusterNode *_thisNode;
	int _clusterSize;
	
public:
	ClusterRandomScheduler();
	~ClusterRandomScheduler();
	
	ComputePlace *addReadyTask(Task *task, ComputePlace *hardwarePlace, ReadyTaskHint hint, bool doGetIdle = true);
	
	Task *getReadyTask(ComputePlace *hardwarePlace, Task *currentTask = nullptr, bool canMarkAsIdle = true, bool doWait = false);
	
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


#endif // CLUSTER_RANDOM_SCHEDULER_HPP

