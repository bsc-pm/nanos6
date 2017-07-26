/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "SchedulerGenerator.hpp"

bool SchedulerGenerator::_collapsable = false;

// Get the Host scheduler
// This is the scheduler that is called through the Scheduler class. Therefor, this is the initializer
SchedulerInterface *SchedulerGenerator::createHostScheduler()
{
	EnvironmentVariable<std::string> schedulerName("NANOS6_SCHEDULER", "default");

	if (schedulerName.getValue() == "hierarchical") {
		return new HostHierarchicalScheduler();
	} else if (schedulerName.getValue() == "collapsable") {
		_collapsable = true;
		
		SchedulerInterface *scheduler = nullptr;
		
		// Check if this scheduler level can be collapsed
		if (HostHierarchicalScheduler::canBeCollapsed()) {
			scheduler = createNUMAScheduler();
		} else {
			scheduler = new HostHierarchicalScheduler();
		}

		return scheduler;
	} else {
		return createCPUScheduler(schedulerName.getValue(), -1);
	}
}
