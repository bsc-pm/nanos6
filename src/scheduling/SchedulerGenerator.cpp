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
		
		SchedulerInterface *scheduler = new HostHierarchicalScheduler();
		
		// Check if this scheduler level can be collapsed
		if (scheduler->canBeRemoved()) {
			delete scheduler;
			scheduler = createNUMAScheduler();
		}

		return scheduler;
	} else {
		return createCPUScheduler(schedulerName.getValue());
	}
}
