/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef HOST_SCHEDULER_HPP
#define HOST_SCHEDULER_HPP

#include "HostUnsyncScheduler.hpp"
#include "SyncScheduler.hpp"

class HostScheduler : public SyncScheduler {
public:
	HostScheduler(size_t totalComputePlaces, SchedulingPolicy policy, bool enablePriority, bool enableImmediateSuccessor, bool enableLocality)
		: SyncScheduler(totalComputePlaces)
	{
		_scheduler = new HostUnsyncScheduler(policy, enablePriority, enableImmediateSuccessor, enableLocality);
	}

	inline Task *getReadyTask(ComputePlace *computePlace)
	{
		Task *result = getTask(computePlace);
		assert(result == nullptr || result->getDeviceType() == nanos6_host_device);
		return result;
	}

	inline std::string getName() const
	{
		return "HostScheduler";
	}

private:
	inline ComputePlace *getComputePlace(uint64_t computePlaceIndex) const
	{
		const std::vector<CPU *> &cpus = CPUManager::getCPUListReference();
		return cpus[computePlaceIndex];
	}

	inline bool mustStopServingTasks(ComputePlace *computePlace) const
	{
		CPU *cpu = (CPU *) computePlace;
		assert(cpu != nullptr);

		// Unowned compute places cannot keep scheduling
		if (!cpu->isOwned())
			return true;

		// Check disabling or shutting down status
		return !CPUManager::acceptsWork(cpu);
	}

	inline void postServingTasks(ComputePlace *computePlace, Task *assignedTask)
	{
		if (assignedTask == nullptr) {
			// The compute place stopped serving tasks and it did not get any
			// task for itself, so it stopped because it was disabled or it is
			// an external compute place. In this case, we only need to request
			// a compute place to guarantee that someone is serving tasks
			CPUManager::executeCPUManagerPolicy(computePlace, REQUEST_CPUS, 1);
		} else {
			// The compute place stopped serving tasks and it obtained a task
			// for itself. We should keep a compute place scheduling tasks if
			// possible. Additionally, we request another compute place because
			// it seems that there are ready tasks. In this way, we resume compute
			// places progressively when there is available work
			CPUManager::executeCPUManagerPolicy(computePlace, REQUEST_CPUS, 2);
		}
	}
};

#endif // HOST_SCHEDULER_HPP
