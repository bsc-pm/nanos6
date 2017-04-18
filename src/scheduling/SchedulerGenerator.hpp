#ifndef SCHEDULER_GENERATOR_HPP
#define SCHEDULER_GENERATOR_HPP

#include "schedulers/DefaultScheduler.hpp"
#include "schedulers/DeviceHierarchicalScheduler.hpp"
#include "schedulers/FIFOImmediateSuccessorWithPollingScheduler.hpp"
#include "schedulers/FIFOScheduler.hpp"
#include "schedulers/HostHierarchicalScheduler.hpp"
#include "schedulers/ImmediateSuccessorScheduler.hpp"
#include "schedulers/ImmediateSuccessorWithPollingScheduler.hpp"
#include "schedulers/NUMAHierarchicalScheduler.hpp"
#include "schedulers/PriorityScheduler.hpp"
#include "SchedulerInterface.hpp"

#include "lowlevel/EnvironmentVariable.hpp"

class SchedulerGenerator {
private:
	// The hierarchical scheduler may be collapsable. In this case, if a node
	// from the hierarchy has only one children, it won't be generated.
	static bool _collapsable;

	// Get the CPU scheduler
	static inline SchedulerInterface *createCPUScheduler(std::string const &schedulerName)
	{
		if (schedulerName == "default") {
			return new DefaultScheduler();
		} else if (schedulerName == "fifo") {
			return new FIFOScheduler();
		} else if (schedulerName == "immediatesuccessor") {
			return new ImmediateSuccessorScheduler();
		} else if (schedulerName == "iswp") {
			return new ImmediateSuccessorWithPollingScheduler();
		} else if (schedulerName == "fifoiswp") {
			return new FIFOImmediateSuccessorWithPollingScheduler();
		} else if (schedulerName == "priority") {
			return new PriorityScheduler();
		} else {
			std::cerr << "Warning: invalid scheduler name '" << schedulerName << "', using default instead." << std::endl;
			return new DefaultScheduler();
		}
	}

public:
	// Get the Host scheduler
	// This is the scheduler that is called through the Scheduler class. Therefor, this is the initializer
	static SchedulerInterface *createHostScheduler();

	// Get the scheduler for the NUMA nodes
	static inline SchedulerInterface *createNUMAScheduler()
	{
		SchedulerInterface *scheduler = new NUMAHierarchicalScheduler();
		
		// Check if this scheduler level can be collapsed
		if (_collapsable && scheduler->canBeRemoved()) {
			delete scheduler;
			scheduler = createNUMANodeScheduler();
		}

		return scheduler;
	}
	
	static inline SchedulerInterface *createNUMANodeScheduler()
	{
		SchedulerInterface *scheduler = new DeviceHierarchicalScheduler();
		
		// Check if this scheduler level can be collapsed
		if (_collapsable && scheduler->canBeRemoved()) {
			delete scheduler;
			scheduler = createDeviceScheduler();
		}

		return scheduler;
	}
	
	static inline SchedulerInterface *createDeviceScheduler()
	{
		// TODO: when other devices are introduced, add "type" parameter
		EnvironmentVariable<std::string> schedulerName("NANOS6_CPU_SCHEDULER", "default");
	
		return SchedulerGenerator::createCPUScheduler(schedulerName.getValue());
	}
};


#endif // SCHEDULER_GENERATOR_HPP
