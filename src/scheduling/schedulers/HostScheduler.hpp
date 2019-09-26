/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef HOST_SCHEDULER_HPP
#define HOST_SCHEDULER_HPP

#include "HostUnsyncScheduler.hpp"
#include "SyncScheduler.hpp"


class HostScheduler : public SyncScheduler {
public:
	
	HostScheduler(SchedulingPolicy policy, bool enablePriority, bool enableImmediateSuccessor)
	{
		_scheduler = new HostUnsyncScheduler(policy, enablePriority, enableImmediateSuccessor);
	}
	
	virtual ~HostScheduler()
	{
		delete _scheduler;
	}
	
	Task *getReadyTask(ComputePlace *computePlace, ComputePlace * = nullptr)
	{
		Task *result = getTask(computePlace, nullptr);
		assert(result == nullptr || result->getDeviceType() == nanos6_host_device);
		return result;
	}
	
	inline std::string getName() const
	{
		return "HostScheduler";
	}
};

#endif // HOST_SCHEDULER_HPP
