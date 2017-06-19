#ifndef SCHEDULER_GENERATOR_HPP
#define SCHEDULER_GENERATOR_HPP

#include "schedulers/DefaultScheduler.hpp"
#include "schedulers/DeviceHierarchicalScheduler.hpp"
#include "schedulers/FIFOImmediateSuccessorWithMultiPollingScheduler.hpp"
#include "schedulers/FIFOImmediateSuccessorWithPollingScheduler.hpp"
#include "schedulers/FIFOScheduler.hpp"
#include "schedulers/HostHierarchicalScheduler.hpp"
#include "schedulers/ImmediateSuccessorScheduler.hpp"
#include "schedulers/ImmediateSuccessorWithMultiPollingScheduler.hpp"
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
	static inline SchedulerInterface *createCPUScheduler(std::string const &schedulerName, int nodeIndex)
	{
		if (schedulerName == "default") {
			return new DefaultScheduler(nodeIndex);
		} else if (schedulerName == "fifo") {
			return new FIFOScheduler(nodeIndex);
		} else if (schedulerName == "immediatesuccessor") {
			return new ImmediateSuccessorScheduler(nodeIndex);
		} else if (schedulerName == "iswp") {
			return new ImmediateSuccessorWithPollingScheduler(nodeIndex);
		} else if (schedulerName == "fifoiswp") {
			return new FIFOImmediateSuccessorWithPollingScheduler(nodeIndex);
		} else if (schedulerName == "iswmp") {
			return new ImmediateSuccessorWithMultiPollingScheduler(nodeIndex);
		} else if (schedulerName == "fifoiswmp") {
			return new FIFOImmediateSuccessorWithMultiPollingScheduler(nodeIndex);
		} else if (schedulerName == "priority") {
			return new PriorityScheduler(nodeIndex);
		} else {
			std::cerr << "Warning: invalid scheduler name '" << schedulerName << "', using default instead." << std::endl;
			return new DefaultScheduler(nodeIndex);
		}
	}

public:
	// Get the Host scheduler
	// This is the scheduler that is called through the Scheduler class. Therefor, this is the initializer
	static SchedulerInterface *createHostScheduler();

	// Get the scheduler for the NUMA nodes
	static inline SchedulerInterface *createNUMAScheduler()
	{
		SchedulerInterface *scheduler = nullptr;
		
		// Check if this scheduler level can be collapsed
		if (_collapsable && NUMAHierarchicalScheduler::canBeCollapsed()) {
			scheduler = createNUMANodeScheduler(-1);
		} else {
			scheduler = new NUMAHierarchicalScheduler();
		}

		return scheduler;
	}
	
	static inline SchedulerInterface *createNUMANodeScheduler(int nodeIndex)
	{
		SchedulerInterface *scheduler = nullptr;
		
		// Check if this scheduler level can be collapsed
		if (_collapsable && DeviceHierarchicalScheduler::canBeCollapsed()) {
			scheduler = createDeviceScheduler(nodeIndex);
		} else {
			scheduler = new DeviceHierarchicalScheduler(nodeIndex);
		}

		return scheduler;
	}
	
	static inline SchedulerInterface *createDeviceScheduler(int nodeIndex)
	{
		// TODO: when other devices are introduced, add "type" parameter
		EnvironmentVariable<std::string> schedulerName("NANOS6_CPU_SCHEDULER", "default");
	
		return SchedulerGenerator::createCPUScheduler(schedulerName.getValue(), nodeIndex);
	}
};


#endif // SCHEDULER_GENERATOR_HPP
