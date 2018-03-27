/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "schedulers/DefaultScheduler.hpp"
#include "schedulers/DeviceHierarchicalScheduler.hpp"
#include "schedulers/FIFOImmediateSuccessorWithPollingScheduler.hpp"
#include "schedulers/FIFOScheduler.hpp"
#include "schedulers/HostHierarchicalScheduler.hpp"
#include "schedulers/ImmediateSuccessorScheduler.hpp"
#include "schedulers/ImmediateSuccessorWithPollingScheduler.hpp"
#include "schedulers/NaiveScheduler.hpp"
#include "schedulers/NUMAHierarchicalScheduler.hpp"
#include "schedulers/NoSleepPriorityScheduler.hpp"
#include "schedulers/PriorityScheduler.hpp"

#include "SchedulerGenerator.hpp"


bool SchedulerGenerator::_collapsable = false;


// Get the CPU scheduler
SchedulerInterface *SchedulerGenerator::createCPUScheduler(std::string const &schedulerName, int nodeIndex)
{
	if (schedulerName == "default") {
		return new DefaultScheduler(nodeIndex);
	} else if (schedulerName == "naive") {
		return new NaiveScheduler(nodeIndex);
	} else if (schedulerName == "fifo") {
		return new FIFOScheduler(nodeIndex);
	} else if (schedulerName == "immediatesuccessor") {
		return new ImmediateSuccessorScheduler(nodeIndex);
	} else if (schedulerName == "iswp") {
		return new ImmediateSuccessorWithPollingScheduler(nodeIndex);
	} else if (schedulerName == "fifoiswp") {
		return new FIFOImmediateSuccessorWithPollingScheduler(nodeIndex);
	} else if (schedulerName == "priority") {
		return new PriorityScheduler(nodeIndex);
	} else if (schedulerName == "nosleep-priority") {
		return new NoSleepPriorityScheduler(nodeIndex);
	} else {
		std::cerr << "Warning: invalid scheduler name '" << schedulerName << "', using default instead." << std::endl;
		return new DefaultScheduler(nodeIndex);
	}
}


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


// Get the scheduler for the NUMA nodes
SchedulerInterface *SchedulerGenerator::createNUMAScheduler()
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


SchedulerInterface *SchedulerGenerator::createNUMANodeScheduler(int nodeIndex)
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


SchedulerInterface *SchedulerGenerator::createDeviceScheduler(int nodeIndex)
{
	// TODO: when other devices are introduced, add "type" parameter
	EnvironmentVariable<std::string> schedulerName("NANOS6_CPU_SCHEDULER", "default");
	
	return SchedulerGenerator::createCPUScheduler(schedulerName.getValue(), nodeIndex);
}

