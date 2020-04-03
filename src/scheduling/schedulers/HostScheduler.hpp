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

	HostScheduler(size_t totalComputePlaces, SchedulingPolicy policy, bool enablePriority, bool enableImmediateSuccessor)
		: SyncScheduler(totalComputePlaces)
	{
		_scheduler = new HostUnsyncScheduler(policy, enablePriority, enableImmediateSuccessor);
	}

	Task *getReadyTask(ComputePlace *computePlace)
	{
		Task *result = getTask(computePlace);
		assert(result == nullptr || result->getDeviceType() == nanos6_host_device);
		return result;
	}

	inline std::string getName() const
	{
		return "HostScheduler";
	}
};

#endif // HOST_SCHEDULER_HPP
