/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef NUMA_HIERARCHICAL_SCHEDULER_HPP
#define NUMA_HIERARCHICAL_SCHEDULER_HPP

#include <boost/dynamic_bitset.hpp>

#include "hardware/HardwareInfo.hpp"

#include "../SchedulerInterface.hpp"


class Task;


class NUMAHierarchicalScheduler: public SchedulerInterface {
	std::vector<SchedulerInterface *> _NUMANodeScheduler;
	
	// Use atomics to avoid using a lock
	// Be careful, as std::atomic does not have a copy operation, many vector
	// operations are not usable
	std::vector<std::atomic<int>> _readyTasks;
	
	std::vector<std::atomic<int>> _enabledCPUs;
	
	size_t getAvailableNUMANodeCount();
	
public:
	NUMAHierarchicalScheduler();
	~NUMAHierarchicalScheduler();

	ComputePlace *addReadyTask(Task *task, ComputePlace *hardwarePlace, ReadyTaskHint hint, bool doGetIdle = true);
	
	Task *getReadyTask(ComputePlace *hardwarePlace, Task *currentTask = nullptr, bool canMarkAsIdle = true, bool doWait = false);
	
	ComputePlace *getIdleComputePlace(bool force=false);
	
	void disableComputePlace(ComputePlace *hardwarePlace);
	
	void enableComputePlace(ComputePlace *hardwarePlace);
	
	bool requestPolling(ComputePlace *computePlace, polling_slot_t *pollingSlot, bool canMarkAsIdle = true);
	
	bool releasePolling(ComputePlace *computePlace, polling_slot_t *pollingSlot, bool canMarkAsIdle = true);

	static inline bool canBeCollapsed()
	{
		return (HardwareInfo::getMemoryPlaceCount(nanos6_device_t::nanos6_host_device) == 1);
	}
	
	std::string getName() const;
};


#endif // NUMA_HIERARCHICAL_SCHEDULER_HPP

