/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "lowlevel/FatalErrorHandler.hpp"
#include "schedulers/cluster/ClusterLocalityScheduler.hpp"
#include "schedulers/cluster/ClusterRandomScheduler.hpp"
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
#include "schedulers/PriorityScheduler1.hpp"

#include <config.h>

#if defined(USE_CUDA)
#include "schedulers/cuda/CUDANaiveScheduler.hpp"
#endif

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
	} else if (schedulerName == "priority1") {
		return new PriorityScheduler1(nodeIndex);
	} else if (schedulerName == "nosleep-priority") {
		return new NoSleepPriorityScheduler(nodeIndex);
	} else {
		std::cerr << "Warning: invalid scheduler name '" << schedulerName << "', using default instead." << std::endl;
		return new DefaultScheduler(nodeIndex);
	}
}

/*
 * When CUDA is not available, createCUDAScheduler will return nullptr
 */
SchedulerInterface *SchedulerGenerator::createCUDAScheduler(
	__attribute__((unused)) std::string const &schedulerName,
	__attribute__((unused)) int nodeIndex
) {
#if defined(USE_CUDA)
	if (schedulerName == "default") {
		return new CUDANaiveScheduler(nodeIndex);
	} else if (schedulerName == "naive"){
		return new CUDANaiveScheduler(nodeIndex);
	} else {
		std::cerr << "Warning: invalid scheduler name '" << schedulerName << "', using default instead." << std::endl;
		return new CUDANaiveScheduler(nodeIndex);
	}
#endif
	return nullptr;
}


// Get the Host scheduler
// This is the scheduler that is called through the Scheduler class. Therefor, this is the initializer
SchedulerInterface *SchedulerGenerator::createHostScheduler()
{
	EnvironmentVariable<std::string> schedulerName("NANOS6_SCHEDULER", "default");
	
	if (schedulerName.getValue() == "cluster-random") {
		return new ClusterRandomScheduler();
	} else if (schedulerName.getValue() == "cluster-locality") {
		return new ClusterLocalityScheduler();
	} else if (schedulerName.getValue() == "hierarchical") {
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
		scheduler = createDeviceScheduler(nodeIndex, nanos6_device_t::nanos6_host_device);
	} else {
		scheduler = new DeviceHierarchicalScheduler(nodeIndex);
	}
	
	return scheduler;
}


SchedulerInterface *SchedulerGenerator::createDeviceScheduler(int nodeIndex, nanos6_device_t type)
{
	if (type == nanos6_device_t::nanos6_host_device) {	
		EnvironmentVariable<std::string> schedulerName("NANOS6_CPU_SCHEDULER", "default");
		return SchedulerGenerator::createCPUScheduler(schedulerName.getValue(), nodeIndex);
	} else if (type == nanos6_device_t::nanos6_cuda_device) {
		EnvironmentVariable<std::string> schedulerName("NANOS6_CUDA_SCHEDULER", "default");
		return SchedulerGenerator::createCUDAScheduler(schedulerName.getValue(), nodeIndex);
	} else {
		std::cerr << "Warning: invalid scheduler type '" << type << "', creating host scheduler instead." << std::endl;
		EnvironmentVariable<std::string> schedulerName("NANOS6_CPU_SCHEDULER", "default");
		return SchedulerGenerator::createCPUScheduler(schedulerName.getValue(), nodeIndex);
	}
}

